'''Create a data-base with all trainIDs and their corresponding hdf5 file'''
import os
import h5py
import sqlite3
from sqlite3 import Error
import glob

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def create_files(conn, file):
    """
    Create a new file into the files table
    :param conn: SQLite handle
    :param file: Flash DAQ hdf5 filename
    :return: file id
    """
    sql = ''' INSERT INTO files(name) VALUES(?) '''
    cur = conn.cursor()
    cur.execute(sql, (file,))
    conn.commit()
    return cur.lastrowid

def create_trainids(conn, trainid, file):
    """
    Create a new trainID into the trainIDs table
    :param conn: SQLite handle
    :param trainid: train DAQ hdf5 filename
    :return: trainID id
    """
    sql = f''' INSERT INTO trainIDs(id, file_id)
               VALUES(?, (SELECT id FROM files WHERE name=="{file}")) '''
    cur = conn.cursor()
    trainid = [(int(i),) for i in trainid]
    cur.executemany(sql, trainid)
    conn.commit()
    return cur.lastrowid



def main():
    import numpy as np
    database = 'out/trainIDs.db'

    # create a database connection
    conn = create_connection(database)
    with conn:
        # get all hdf5 files where data from the DAQ is stored
        files = glob.glob('daq/*.h5')
        for file in files[:]:
            print(file)
            fname = os.path.basename(file)
            create_files(conn, fname)
            with h5py.File(file, 'r') as f:
                ids = f['/FL1/Experiment/BL1/ADQ412 GHz ADC/CH00/TD/index'][:]
                create_trainids(conn, ids, fname)


if __name__ == '__main__':
    main()
