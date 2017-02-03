import chatbot
import sys

def main(name,dirname,myname):
    print('Press c to quit interactive mode.')
    bot = chatbot.Chatbot(dirname=dirname)
    bot.initialize()
    while True:
        line = input(myname+': ')
        if line is 'c':
            sys.exit()

        words = bot.get_reply(line)
        words = words.replace('○',myname)
        words = words.replace('〇',myname)
        print(name + ': '+words)

if __name__ == '__main__':
    try:
        name = sys.argv[1]
        dirname = sys.argv[2]
        myname = sys.argv[3]
    except:
        name = 'Alice'
        dirname = '0'
        myname = 'You'
    main(name=name, dirname=dirname, myname=myname)
