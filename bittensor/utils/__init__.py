import getpass

class Cli:
    @staticmethod
    def ask_password():
        print ('here')
        return getpass.getpass("Enter password to unlock key: ")
