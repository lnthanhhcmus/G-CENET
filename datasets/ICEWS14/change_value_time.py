import os

def read_write_file(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])

    with open(os.path.join(inPath, fileName), 'w') as fr:
        for quad in quadrupleList:
            fr.writelines('{}   {}  {}  {}\n'.format(quad[0], quad[1], quad[2], quad[3] - 1))

 
read_write_file("", 'test.txt')