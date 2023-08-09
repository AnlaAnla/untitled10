from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii

# 生成RSA密钥
keyPair = RSA.generate(bits=1024)
pubKey = keyPair.publickey()
pubKeyPEM = pubKey.exportKey()
privKeyPEM = keyPair.exportKey()

print(f"Public key:  {pubKeyPEM.decode('ascii')}")
print(f"Private key: {privKeyPEM.decode('ascii')}")

# 使用公钥进行加密
def encrypy(msg):
    msg = msg.encode('utf-8')
    encryptor = PKCS1_OAEP.new(pubKey)
    encrypted = encryptor.encrypt(msg)
    print("Encrypted:", binascii.hexlify(encrypted))
    return encrypted

# 使用私钥进行解密
def decrypt(encrypted):
    decryptor = PKCS1_OAEP.new(keyPair)
    decrypted = decryptor.decrypt(encrypted)
    print('Decrypted:', decrypted.decode('utf-8'))
    return decrypted

msg = '它是目前最重要的加密算法！计算机通信安全的基'

secret = encrypy(msg)

