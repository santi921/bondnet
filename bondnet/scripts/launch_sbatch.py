import os 

def launch_all():
    """Launch all .sh jobs in folders containing settings.txt files"""
    for root, dirs, files in os.walk("."):
        if "settings.txt" in files:
            os.chdir(root)
            os.system("sbatch *.sh")
            os.chdir("..")

