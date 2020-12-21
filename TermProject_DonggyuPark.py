#!/usr/bin/env python
# coding: utf-8

# In[29]:


area = [
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0]
]
x = 42

section = [[0 for _ in range(x)] for _ in range(x)]
for i in range(1,x-1):
    for j in range(1,x-1):
        if i < x//3 *2 and j < x//3:
            section[i][j] = 1
        elif i>=x//3 *2:
            section[i][j] = 1
for i in section:
    print(i)


# In[30]:


cnt = 0
for line in area:
    for i in range(len(line)):
        if line[i]:
            line[i] = cnt + 1
            cnt += 1
area

cnt = 0
for line in section:
    for i in range(len(line)):
        if line[i]:
            line[i] = cnt + 1
            cnt += 1
for i in section:
    print(i)
print(cnt)


# In[18]:


#초기 빈 행렬식 A
# cnt가 미지수의 개수
A = [[0 for _ in range(cnt+1)] for _ in range(cnt)]
# for i in A:
#     print(i)


# In[31]:


dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1] #이동기
def solve(y, x, linenum):
    for k in range(4):
        next_y = y + dy[k]
        next_x = x + dx[k]
        if 0 <= next_x < len(section) and 0 <= next_y < len(section) and section[next_y][next_x]:
            nodenum = section[next_y][next_x]-1
            A[linenum][nodenum] = -1
for y in range(len(section)):
    for x in range(len(section)):
        if section[y][x]:
            linenum = section[y][x]-1  
            A[linenum][linenum] = 4
            solve(y,x,linenum)
for line in A:
    line[-1] = 0.005
# for line in A:
#     print(line)
            


# In[32]:


import numpy as np
n=cnt
A = np.array(A, dtype=float)
pivot_target = []
for i in range(0, n-1):
    for j in range(i+1, n):
        temp = A[j,i]/A[i,i]
        A[j,:] = A[j,:] - temp * A[i,:]
#     print(np.matrix(A))
        
    # 대각행렬에 0이 있는 행은 pivot_target이라는 리스트에 추가.
    for x in range(i,n-1):
        if not A[x,x]:
            pivot_target.append(x)
            
    #pivot_target이 다 떨어질 때까지 반복하면서 피버팅
    while pivot_target:
#         print("While 안에 들어옴")
#         print(np.matrix(A))
        x = pivot_target.pop()
        for change_target in range(i+1, n-1):
            if A[x,change_target] and A[change_target,x]:
                A[[x,change_target]] = A[[change_target,x]]
x = np.zeros(shape=(n,1), dtype=float)
for i in range(n-1,-1,-1):
    if i==n-1:
        x[i]=A[i,n]/A[i,i]
    else:
        x[i]=A[i,n]/A[i,i]
        for j in range(1, n-i):
            x[i] = x[i]- (A[i,i+j]/A[i,i])*x[i+j]
print(x)


# In[21]:


xx = []
for i in range(len(x)):
    xx.append(float(x[i]))
xx


# In[23]:


potential_section = [[0 for _ in range(len(section))] for _ in range(len(section))]
cnt = 0
for i in range(len(section)):
    for j in range(len(section)):
        if section[i][j]:
            potential_section[i][j] = xx[cnt]
            cnt += 1
potential_section


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

H = np.array(potential_section)

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


# In[26]:


h = 0.003/42
tauzy =  [[0 for _ in range(len(section))] for _ in range(len(section))]
tauzx =  [[0 for _ in range(len(section))] for _ in range(len(section))]
for y in range(len(potential_section)):
    for x in range(1,len(potential_section)-1):
        prev_x = x-1
        next_x = x+1
        if not potential_section[y][prev_x]:
            tauzy[y][x] = (potential_section[y][x]-potential_section[y][prev_x])/h
        elif not potential_section[y][next_x]:
            tauzy[y][x] = (potential_section[y][x]-potential_section[y][next_x])/h
        else:
            tauzy[y][x] = (potential_section[y][next_x]-potential_section[y][prev_x])/(2*h)
H = np.array(tauzy)

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


# In[27]:



for y in range(1,len(potential_section)-1):
    for x in range(1,len(potential_section)-1):
        prev_y = y-1
        next_y = y+1
        if not potential_section[prev_y][x]:
            tauzx[y][x] = (potential_section[y][x]-potential_section[prev_y][x])/h
        elif not potential_section[y][next_x]:
            tauzx[y][x] = (potential_section[y][x]-potential_section[next_y][x])/h
        else:
            tauzx[y][x] = (potential_section[next_y][x]-potential_section[prev_y][x])/(2*h)
H = np.array(tauzx)

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


# In[28]:


shear_stress = [[0 for _ in range(len(section))] for _ in range(len(section))]
for y in range(1,len(potential_section)-1):
    for x in range(1,len(potential_section)-1):
        if tauzy[y][x] or tauzx[y][x]:
            shear_stress[y][x] = pow(tauzy[y][x]**2 + tauzx[y][x]**2, 0.5)
H = np.array(shear_stress)

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


# In[ ]:




