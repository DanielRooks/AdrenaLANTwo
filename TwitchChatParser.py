import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import re
import csv
import sys

def main(argv):
    df = pd.DataFrame(columns=['nickname', 'msg'])
    f = open(argv[0],'r', encoding="utf8") 

    for line in f:
        #separate all the lines
        if not len(line.strip()) == 0 :
            line = line.rstrip()
            line = line.split('] ')
            line[0]+=']'
            
            #time=line[0]

            nickandmsg = line[1]
            nickandmsg = nickandmsg.split(':')
            
            # print(time + str(nickandmsg[0]) + str(nickandmsg[1]))
            #'time': time, 
            data = {'nickname' : nickandmsg[0], 'msg': nickandmsg[1]}
            df = df.append(data, ignore_index=True)
    
    df.to_csv('data.csv', encoding='utf-8', index=False)

  


if __name__ == '__main__':
    main(sys.argv[1:])