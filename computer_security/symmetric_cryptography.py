from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import scrypt
from Crypto.Cipher import AES
import os


filename = "binData.bin"
podaci=""
key=""

while True:
    user_input = input()
    splitano = user_input.split(" ")
    if user_input == "help":
        print("There are 4 things you can do with this program, here is the format:")
        print("1)Initialization-> init [masterPassword] -> initializes your master password")
        print("2)Storing-> put [masterPassword] [address] [password] -> stores an address-password pair")
        print("3)Retrieving-> get [masterPassword] [address] -> retrieves a password of an address")
        print("4)Exiting-> quit\n")
        print("If there is a problem with your command then you will get an error message.")
        print("If there is no address in the system you will get a warning.\n")

    elif user_input == "quit":
        break
    elif len(splitano)==2 and splitano[0]=="init":
        open(filename, 'wb').close()
        print("Password manager initialized.")
        
        
        podaci="prazno"
        i=open(filename, 'ab')
        salt = get_random_bytes(32)
        i.write(salt)
        key=scrypt(splitano[1], salt, key_len=32, N=2**17, r=8, p=1)

        cipher = AES.new(key, AES.MODE_GCM)
        i.write(cipher.nonce)
        encrypted_data = cipher.encrypt(podaci.encode('utf8'))
        i.write(encrypted_data)
        i.write(cipher.digest())
        i.close()
        

    elif len(splitano)==4 and splitano[0]=="put" and os.path.isfile("binData.bin"):

        
        dp=open(filename, 'rb')
        salt = dp.read(32)
        key = scrypt(splitano[1], salt, key_len=32, N=2**17, r=8, p=1)
        nonce = dp.read(16)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

        dp_size = os.path.getsize(filename)
        encrypted_data_size = dp_size - 32 - 16 - 16 

        data = dp.read(encrypted_data_size)
        decrypted_data = cipher.decrypt(data)

        output = True

        tag = dp.read(16)
        try:
            cipher.verify(tag)
        except ValueError:
            print("Master password incorrect or integrity check failed.")
            decrypted_data=b''
            output=False
            dp.close()

        dp.close()

        if(output):
            podaci=decrypted_data.decode('utf8')

            f = open(filename, 'wb')
            salt = get_random_bytes(32) #32*8=256 bita
            f.write(salt)

            key = scrypt(splitano[1], salt, key_len=32, N=2**17, r=8, p=1)
       
            adresa = splitano[2]
            lozinka = splitano[3]

            if podaci=="prazno":
                podaci= adresa + " " + lozinka + " "

            else:
                lista = podaci.split(" ")
                del lista[len(lista)-1]
                replace=False
                for i in range(len(lista)-1):
                    if(lista[i]==adresa):
                        replace=True
                        indexes=[i,i+1]
                        for index in sorted(indexes, reverse=True):
                            del lista[index]
                        podaci=" ".join(lista) + " "
                        print("Changing password...")
                        break
                
                
                podaci+= adresa + " " + lozinka + " "
                         

            cipher = AES.new(key, AES.MODE_GCM)
            f.write(cipher.nonce)
            encrypted_data = cipher.encrypt(podaci.encode('utf8'))
            f.write(encrypted_data)
            f.write(cipher.digest())

            print("Stored password for {}.".format(adresa))

            f.close()

    elif len(splitano)==3 and splitano[0]=="get" and os.path.isfile('binData.bin'):
        ff = open(filename, 'rb')
        salt = ff.read(32)
        key = scrypt(splitano[1], salt, key_len=32, N=2**17, r=8, p=1)
        nonce = ff.read(16)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

        ff_size = os.path.getsize(filename)
        encrypted_data_size = ff_size - 32 - 16 - 16  # Total - salt - nonce - tag = encrypted data

        data = ff.read(encrypted_data_size)
        decrypted_data = cipher.decrypt(data)

        output = True

        tag = ff.read(16)
        try:
            cipher.verify(tag)
        except ValueError:
            print("Master password incorrect or integrity check failed.")
            decrypted_data=b''
            output=False
            ff.close()
        
        ff.close()

        if(output):
            
            adresa = splitano[2]
            podaci = decrypted_data.decode('utf8')
            lista = podaci.split(" ")
            while("" in lista):
                lista.remove("")
            noSuchAddress = True
            for i in range(len(lista)):
                if(lista[i]==adresa and i%2==0):
                    noSuchAddress = False
                    print("Password for {} is: {}".format(adresa, lista[i+1]))
                    break
            if noSuchAddress:
                print("WARNING: There is no such address.")
    else:
        print("ERROR: There is a problem. Type help for additional help.")



