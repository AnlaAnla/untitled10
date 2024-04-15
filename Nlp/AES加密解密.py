from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# 密钥长度必须为16、24或32字节
key = b'0123456789abcdef'  # 16字节密钥

# 需要加密的数据
data = b'This is a secret message'

# 加密数据
cipher = AES.new(key, AES.MODE_ECB)
encrypted_data = cipher.encrypt(pad(data, AES.block_size))
print("Encrypted data:", encrypted_data)

# 解密数据
cipher = AES.new(key, AES.MODE_ECB)
decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
print("Decrypted data:", decrypted_data)
