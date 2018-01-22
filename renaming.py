import os

for num in range(13, 283):
    n = num+1
    print(n)
    os.rename('Train14/H2N2A/frame'+str(n)+'.jpg', 'Train14/H2N2A/frame'+str(num)+'.jpg')
