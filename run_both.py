import subprocess

# Run the first script
result = subprocess.run(["python", r"F:\Omar 3amora\Photo Booth\MODNet\run_prog.py"])

# If the first script ran successfully (exit code 0)
if result.returncode == 0:
    print(" run_prog.py finished successfully, now running try.py...")
    subprocess.run(["python", r"F:\Omar 3amora\Photo Booth\Harmonizer\try.py"])
else:
    print(" run_prog.py failed. try.py will not run.")
