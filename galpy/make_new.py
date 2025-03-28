import os
import galpy

install_dir = os.path.split(galpy.__file__)[0]

def main():

    # Copy starting files to new analysis directory:
    new_dir = os.getcwd()
    os.system("scp %s/run_galmaps.py %s" %(install_dir,new_dir))

########################
if __name__=="__main__":
        main(sys.argv)

