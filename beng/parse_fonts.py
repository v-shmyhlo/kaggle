import urllib.request

lines = open('./beng/fonts/fonts.txt').readlines()

paths = lines[6::9]

paths = [p.split(',')[2][5:-19] for p in paths]

for i, p in enumerate(paths):
    urllib.request.urlretrieve(p, './beng/fonts/{}.woff2'.format(i))
