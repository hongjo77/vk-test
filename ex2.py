
def co():
    s=0
    for i in range(1,99999999,s+1):
        count=0
        for j in range(2,i):
            if(i%j==0):
                count+=1
            if(count>5000000):
                print(i)
                return
        s+=1

    
