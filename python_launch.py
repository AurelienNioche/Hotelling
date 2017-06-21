from tqdm import tqdm

from prepare import main as prepare
from count_args import main as count_args
from script_to_run import main as run


def main():

    prepare()
    n_args = count_args()
    for i in tqdm(range(n_args)):
        run(i)


if __name__ == "__main__":

    main()
