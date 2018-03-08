import sys

r= open('coordinates.txt').read()

r=r.replace(' ','')
fout= open('coordinates.txt', 'w')
fout.write(r)

fout.close()
