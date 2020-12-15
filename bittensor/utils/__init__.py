import getpass

class Cli:
    @staticmethod
    def ask_password():
        return getpass.getpass("Enter password to unlock key: ")
