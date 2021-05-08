import numpy as np

from pipeline.centroiding_steps.Centroiding2D import Centroiding2D


class TopologicalPeakDetectionCentroiding(Centroiding2D):

    def __init__(self, *args, **kwargs):
        super(TopologicalPeakDetectionCentroiding, self).__init__(*args, **kwargs)

    @staticmethod
    def __get(im, p):
        return im[p[0]][p[1]]

    @staticmethod
    def __iter_neighbors(p, w, h):
        y, x = p

        # 8-neighborship
        neigh = [(y + j, x + i) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        # 4-neighborship
        # neigh = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]

        for j, i in neigh:
            if j < 0 or j >= h:
                continue
            if i < 0 or i >= w:
                continue
            if j == y and i == x:
                continue
            yield j, i

    @staticmethod
    def __persistence(image_2D):
        h, w = image_2D.shape

        # Get indices orderd by value from high to low
        indices = [(i, j) for i in range(h) for j in range(w)]
        indices.sort(key=lambda p: TopologicalPeakDetectionCentroiding.__get(image_2D, p), reverse=True)

        # Maintains the growing sets
        uf = UnionFind()

        groups0 = {}

        def get_comp_birth(p):
            return TopologicalPeakDetectionCentroiding.__get(image_2D, uf[p])

        # Process pixels from high to low
        for i, p in enumerate(indices):
            v = TopologicalPeakDetectionCentroiding.__get(image_2D, p)
            ni = [uf[q] for q in TopologicalPeakDetectionCentroiding.__iter_neighbors(p, w, h) if q in uf]
            nc = sorted([(get_comp_birth(q), q) for q in set(ni)], reverse=True)

            if i == 0:
                groups0[p] = (v, v, None)

            uf.add(p, -i)

            if len(nc) > 0:
                oldp = nc[0][1]
                uf.union(oldp, p)

                # Merge all others with oldp
                for bl, q in nc[1:]:
                    if uf[q] not in groups0:
                        # print(i, ": Merge", uf[q], "with", oldp, "via", p)
                        groups0[uf[q]] = (bl, bl - v, p)
                    uf.union(oldp, q)

        groups0 = [(k, groups0[k][0], groups0[k][1], groups0[k][2]) for k in groups0]
        groups0.sort(key=lambda g: g[2], reverse=True)

        return groups0

    def find_centres(self, image_2D):
        g0 = TopologicalPeakDetectionCentroiding.__persistence(image_2D)
        points = []
        for i, homclass in enumerate(g0):
            p_birth, bl, pers, p_death = homclass
            if pers > 200:
                x, y = p_birth
                points.append([x, y])
                
        return np.array(points)


class UnionFind:
    """Union-find data structure.

    Union-find data structure. Based on Josiah Carlson's code,
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
    with significant additional changes by D. Eppstein.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def add(self, object, weight):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = weight

    def __contains__(self, object):
        return object in self.parents

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            assert(False)
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure.

        """
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.parents[r] = heaviest
