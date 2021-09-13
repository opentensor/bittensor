import getpass

class Cli:
    """ Allow users to enter password through command line 
    """
    @staticmethod
    def ask_password():
        return getpass.getpass("Enter password to unlock key: ")
