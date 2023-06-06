import sys
import re
import getpass
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import scrypt

filename="something.bin"
pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^a-zA-Z\d]).{8,}$'

if(len(sys.argv)!=3):
    print("Pogreška")
    sys.exit(1)
action = sys.argv[1]
username = sys.argv[2]
if(action=="add"):
    password1 = getpass.getpass("Password:")
    password2 = getpass.getpass("Repeat password:")
    if(password1!=password2):
        print("User add failed. Password mismatch.")
    elif not re.match(pattern, password1):
        print("Not matching regular expression.")
    else:
        username_bytes = username.encode("utf-8")
        f=open(filename, 'ab')
        salt = get_random_bytes(32)
        hash=scrypt(password1, salt, key_len=32, N=2**17, r=8, p=1)
        f.write(username_bytes)
        f.write("š".encode("utf-8"))
        f.write(salt)
        f.write("š".encode("utf-8"))
        f.write(hash)
        f.write("š".encode("utf-8"))
        f.write("0š".encode("utf-8"))
        f.close()
        print("User {} successfuly added.".format(username))
elif(action=="passwd"):
    password1 = getpass.getpass("Password:")
    password2 = getpass.getpass("Repeat password:")
    if(password1!=password2):
        print("Password changed failed. Password mismatch.")
    elif not re.match(pattern, password1):
        print("Not matching regular expression for password. The password has to be at least 8 characters long, have both upper and lower letters, have at least one number and one special character.")
    else:
        f=open(filename, "rb")
        line = f.read()
        parts = line.split("š".encode("utf-8"))
        for i in range(len(parts)):
            if(i%4==0):
                user = parts[i].decode("utf-8")
                salt = parts[i+1]
                hash = parts[i+2]
                flag = parts[i+3].decode("utf-8")
                if(user==username):
                    new_salt = get_random_bytes(32)
                    new_hash = scrypt(password1, new_salt, key_len=32, N=2**17, r=8, p=1)
                    parts[i+1] = new_salt
                    parts[i+2] = new_hash
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
elif(action=="forcepass"):
    f=open(filename, "rb")
    line = f.read()
    parts = line.split("š".encode("utf-8"))
    for i in range(len(parts)):
        if(i%4==0):
            user = parts[i].decode("utf-8")
            flag = parts[i+3].decode("utf-8")
            new_flag = "1"
            if(user==username):
                parts[i+3] = new_flag.encode("utf-8")
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
    print("User will be requested to change password on next login.")
elif(action=="del"):
    f=open(filename, "rb")
    line = f.read()
    parts = line.split("š".encode("utf-8"))
    for i in range(len(parts)):
        if(i%4==0):
            user = parts[i].decode("utf-8")
            if(user==username):
                parts[i]=b''
                parts[i+1] = b''
                parts[i+2] = b''
                parts[i+3] = b''
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

    print("User successfully removed.")
    
else:
    print("Pogreška")

