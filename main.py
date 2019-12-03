import getopt
import sys

from colorama import Fore

from models.mle.Mle import Mle


def set_gan(gan_name):
    gans = dict()
    gans['mle'] = Mle
    Gan = gans[gan_name.lower()]
    gan = Gan()
    gan.vocab_size = 5000
    gan.generate_num = 10000
    return gan
    

def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hg:t:d:")

        opt_arg = dict(opts)
        
        gan = set_gan('mle')
        gan.train_oracle()
        gan_func()
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    gan = None
    parse_cmd(sys.argv[1:])
