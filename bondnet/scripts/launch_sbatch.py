import os 

def launch_all():
    """Launch all .sh jobs in folders containing settings.txt files"""
    for root, dirs, files in os.walk("."):
        condition = False
        for i in files: 
            #print(i)
            if("settings" in i): condition = True
        if condition:
            #print(files)
                #print(root)
                os.chdir(root)
                os.system("sbatch *.sh")
                os.chdir("..")
launch_all()
