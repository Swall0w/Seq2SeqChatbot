import MeCab
import sys

def main():
    index = 0
    while True:
        filename = 'data/' +str(index) + '/cat.txt'
        sentence = []
        try:
            with open(filename) as f:
                line = f.readline()
                while line:
                    sentence.append(line.rstrip('\n'))
                    line = f.readline()

            whitefile = open('data/separate/white.txt',mode='a',encoding='utf-8')
            with whitefile:
                print(len(sentence))
                if len(sentence)%2 is 1:
                    sentence = sentence[:-1]
                    print(len(sentence))

                for row, item in enumerate(sentence):
                    result = []
                    if row %2 is 0:
                        result.append(item)
                    else :
                        result.append('\t')
                        result.append(item)
                        result.append('\n')
                    whitefile.write(''.join(result))
            index += 1
        except:
            sys.exit()
            print('end')


if __name__ == '__main__':
    main()
