#!/bin/python3

from argparse import ArgumentParser
from pathlib import Path
from loguru import logger

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

test_json = """{ 
  "accountId": "0x96f84a32fe0fd3b5e61b68705340e50a300562a85172b97ec113e76665caaf65",
  "publicKey": "0x96f84a32fe0fd3b5e61b68705340e50a300562a85172b97ec113e76665caaf65",
  "secretPhrase": "duty embrace auto sketch ring fluid enough resemble insect nuclear top vital congress ship conduct lobster lunch pause clump walk wash stereo force scrap",
  "secretSeed": "0xde2c5031d26e80c9b553b94fc7a6ba39b4b83e103689e2bfc425104d6dcaf35c",
  "ss58Address": "5FUeoCZBwqh7bNsMB73iDDAQhmWjjzYAMkK1vUP6jXe172yp"
}"""



def generate_key(password):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), salt=b"Iguesscyborgslikemyselfhaveatendencytobeparanoidaboutourorigins", length=32, iterations=10, backend=default_backend())
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


parser = ArgumentParser(description="Key loading test script")
parser.add_argument("--key", required=False, default="~/.bittensor/key", help="Path to the keyfile")

args = parser.parse_args()
file = Path(args.key)
file = file.expanduser()

if not file.is_file():
    logger.error("File {} not found.", file.__fspath__())


password = input("Enncryption password ?")
key = generate_key(password)





clear = base64.urlsafe_b64decode(cipher_text)
print(clear)

password = input('Decryption password ?')
key = generate_key(password)

cipher_suite = Fernet(key)
plaintext = cipher_suite.decrypt(cipher_text)


print(plaintext.decode())










