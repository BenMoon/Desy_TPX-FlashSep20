/* just copy paste this into the sqlite CLI
   alternatively "sqlite3 < create_trainID_lookup.sql" should work as well
 */
CREATE TABLE IF NOT EXISTS files (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trainIDs (
    id INTEGER NOT NULL PRIMARY KEY,
    file_id integer NOT NULL,
    FOREIGN KEY (file_id) REFERENCES files (id)
);
/*
CREATE INDEX trainIDs_idx ON trainIDs (id);

INSERT INTO files (name) VALUES ('testfile.h5');
INSERT INTO trainIDs (train_id, file_id) VALUES (123, (select file_id from files where name=='testfile.h5'));

SELECT * FROM trainIDs WHERE trainIDs.id = 842098343;
SELECT * FROM files WHERE id IN (SELECT file_id FROM trainIDs WHERE id = 842098343 OR id = 843723669);
SELECT * FROM files WHERE id IN (SELECT file_id FROM trainIDs WHERE (id > 842098343 AND id < 842098353) OR id = 843723669);
 */
