import pyodbc
from keras.models import load_model
import numpy as np
from sklearn.externals import joblib
import sys

# Load the trained Preprocessing data scaler
scaler_filename = 'scaler.save'
scaler = joblib.load(scaler_filename)


# Process data
def process(ls, ignore_cols):
    for id in sorted(ignore_cols, reverse=True):
        [ls.pop(id)]
    for i in range(6):
        ls[1] = 1 if ls[1] == 'Female' else 0
    _input = np.reshape(ls, (1, 6))
    _input = scaler.transform(_input)
    return _input

# Load trained Model
model = load_model('model.h5')

# ODBC
file_path = r"""C:\Users\jackh\Documents\Comp_Tools_for_eng\TitanicFinal\Titanic.accdb"""
user = "admin"
password = ""
odbc_conn_str = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;UID=%s;PWD=%s' % \
                (file_path, user, password)

conn = pyodbc.connect(odbc_conn_str)
crsr = conn.cursor()
sql_statement = "select * from Python where survived is null"
crsr.execute(sql_statement)
results = crsr.fetchall()


# Looks up which records have null as survived field and calculates pred
for record in results:
    record = [x for x in record]
    id = record[0]
    name = record[2]
    ignore = [-1, 0, 2]
    pred = model.predict(process(record, ignore))
    sql = 'UPDATE Python SET survived = ' + str(pred[0][0]) + ' WHERE ID = ' + str(id)
    print(sql)
    crsr.execute(sql)
    conn.commit()


# Closes the ODBC connection
crsr.close()
conn.close()
sys.exit()
