import subprocess


def call_command_line(string, **kwargs):
    """Executes string as a command line prompt. stdout and stderr are keyword args."""
    return subprocess.run(string.split(" "), **kwargs)
