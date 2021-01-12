import os
import sys
import subprocess

# def os_system_wrapper(cmd):
#    p = subprocess.Popen(['osascript', '-'], stdin=subprocess.PIPE, universal_newlines=True)
#    p.communicate(cmd)

# cmd = """tell application "Terminal" to activate do script "/bin/bash -c 'eval " & quote & "$(~/Documents/flameTimewarpML/miniconda3/bin/conda shell.bash hook)" & quote & "; conda activate; sleep 5'; exit"  """
# subprocess.Popen(['osascript', '-e', cmd])
logfile_path = '/var/tmp/flameTimewarpML_install.log'
log_cmd = "konsole -e /bin/bash -c 'tail -f " + os.path.abspath(logfile_path) +"'"
print (log_cmd)
os.system(log_cmd)
# subprocess.Popen(['konsole', '-e', log_cmd])