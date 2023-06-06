import sys
import getpass
from Crypto.Protocol.KDF import scrypt
from Crypto.Random import get_random_bytes

filename="something.bin"

if(len(sys.argv)!=3):
    print("Pogreška")
    sys.exit(1)
action = sys.argv[1]
username = sys.argv[2]
if(action!="login"):
    print("Pogreška")
    sys.exit(1)
password1 = getpass.getpass("Password:")
f = open(filename, "rb")
line = f.read()
parts = line.split("š".encode("utf-8"))
found=False
forced=False
for i in range(len(parts)):
    if(i%4==0):
        if(parts[i].decode("utf-8")==username):
            flag=parts[i+3].decode("utf-8")
            if(flag=="1"):
                found = True
                salt = parts[i+1]
                old_hash = parts[i+2]
                new_hash = scrypt(password1, salt, key_len=32, N=2**17, r=8, p=1)
                if new_hash!=old_hash:
                    print("Username or password incorrect.")
                    break                  
                password1_new = getpass.getpass("New password:")
                password2_new = getpass.getpass("Repeat new password:")
                if(password1_new!=password2_new):
                    print("User add failed. Password mismatch.")
                    break
                elif(scrypt(password1_new, salt, key_len=32, N=2**17, r=8, p=1)==old_hash):
                    print("Old password can not be new password.")
                    break
                else:
                    forced=True
                    new_salt_forced = get_random_bytes(32)
                    new_hash_forced=scrypt(password1_new, new_salt_forced, key_len=32, N=2**17, r=8, p=1)
            elif(flag=="0"):    
                found=True
                salt = parts[i+1]
                old_hash = parts[i+2]
                new_hash = scrypt(password1, salt, key_len=32, N=2**17, r=8, p=1)
                if new_hash==old_hash:
                    print("Login successful.")
                else:
                    print("Username or password incorrect.")
                break
if not found:
    print("Username or password incorrect.")
else:
    if forced:
        f=open(filename, "rb")
        line = f.read()
        parts = line.split("š".encode("utf-8"))
        for i in range(len(parts)):
            if(i%4==0):
                user = parts[i].decode("utf-8")
                if(user==username):
                    parts[i+1] = new_salt_forced
                    parts[i+2] = new_hash_forced
                    parts[i+3] = "0".encode("utf-8")
                    break
        new = b''
        for part in parts:
            if part==b'':
                continue
            new+=part
            new+="š".encode("utf-8")
        f.close()
        f=open(filename, "wb")
        f.write(new)
        f.close()
        print("Password change successful.")

