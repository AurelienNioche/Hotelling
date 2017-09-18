from subprocess import getoutput, Popen
from os import path
from time import sleep
import sys


def main(server_folder="aurelien/Hotelling",
         local_folder=path.expanduser("~/Desktop")):

    cmd = """expect -c 'spawn avakas
             expect "~]$"
             send "echo separator > /dev/null \r"
             expect "~]$"
             send "qstat -u anioche && exit 0\r"
             interact'"""

    # keywords appearing in qstat cmd if jobs are running
    keywords = ["master", "ubx2"]

    # nb of second before refreshing
    refresh = 60

    # waiting for jobs to end
    while True:

        print("Looking for data in '{}'...".format(server_folder))

        # connect to avakas and look for jobs
        out = getoutput(cmd).split("separator")[1]

        # if keywords are in output (meaning jobs are finished)
        if all([w not in out for w in keywords]):
            print("Downloading files in '{}'...".format(local_folder))
            cmd2 = "pull_avakas {} {}".format(server_folder, local_folder)
            print("Launch command: '{}'.".format(cmd2))
            print(getoutput(cmd2))
            print("Done.")
            getoutput("say 'Work is done!'")
            break

        print("Jobs seem to be not terminated. I will retry soon.")
        sleep(refresh)


if __name__ == '__main__':
    main(*sys.argv[1:])
