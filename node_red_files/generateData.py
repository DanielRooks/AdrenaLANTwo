import random

i = 0
mood = 0
pre = 0
content = "F"
F = open("D:\\Adrenalan2\\input.csv","w+")
for i in range(1000):

    pre = mood

    if mood < 100:
        mood = mood + random.randint(0, 1)
    if mood < 70:
        mood = mood + random.randint(0, 1)
    if mood < 30:
        mood = mood - random.randint(0, 1)
    if mood > 0:
        mood = mood - random.randint(0, 1)

    if pre < mood:
        content = "F"
    else:
        content = "GG"

    F.write("dyde, " + content + ", " + str(mood) + "\n")

F.close()