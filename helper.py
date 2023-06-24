
import numpy as np
import cv2
from scipy import ndimage

def getEdgePoints(im1,im2):
  im1_pts = np.nonzero(im1)
  im2_pts = np.nonzero(im2)
  return im1_pts, im2_pts

def getInitialTfm(x1,y1,x2,y2):

  x1_mean,y1_mean = np.mean(x1), np.mean(y1)
  x2_mean,y2_mean = np.mean(x2), np.mean(y2)

  x1_var, y1_var = np.var(x1), np.var(y1)
  x2_var, y2_var = np.var(x2), np.var(y2)

  scale_x = x1_var/x2_var
  scale_y = y1_var/y2_var


  trans_x = x1_mean - x2_mean
  trans_y = y1_mean - y2_mean

  return trans_x, trans_y, scale_x, scale_y


def getMatchingPts(x1,y1,x2,y2):

  match_pts = np.empty((x1.shape[0],2))

  for i in range(x1.shape[0]):
    dist = np.inf
    for j in range(x2.shape[0]):
      if 0 < i < x1.shape[0] and 0 < j < x2.shape[0]: 
        temp_dist = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
        if temp_dist < dist:
          dist = temp_dist
          match_pts[i] = np.array([x2[j], y2[j]])

  return match_pts

def constructA(x,y):
  A = np.empty((2*x.shape[0],6))

  for i in range(A.shape[0]):
      j = i//2
      if i%2 == 0:
        A[i,0] = x[j]
        A[i,1] = y[j]
        A[i,2] , A[i,3], A[i,5]= 0,0,0
        A[i,4] = 1
      else:
        A[i,2] = x[j]
        A[i,3] = y[j]
        A[i,0] , A[i,1], A[i,4]= 0,0,0
        A[i,5] = 1
  return A

def constructb(pts):
  b = np.empty((2*pts.shape[0],1))
  for i in range(b.shape[0]):
    j = i//2
    if i%2 == 0:
      b[i] = pts[j,0]
    else:
      b[i] = pts[j,1]
  return b


def align_shape(im1, im2):
  '''
  im1: input edge image 1
  im2: input edge image 2

  Output: transformation T [3] x [3]
  '''
  im1_pts, im2_pts = getEdgePoints(im1,im2)

  x1 = im1_pts[0]
  y1 = im1_pts[1]
  #img1 non zero points 2 X N
  im1_xy = np.vstack((x1,y1))
  x2 = im2_pts[0]
  y2 = im2_pts[1]
  #img2 non zero points 2 X N
  im2_xy = np.vstack((x2,y2))
  iterations = 25

  tx, ty, sx, sy = getInitialTfm(x1,y1,x2,y2)
  m1 = 1
  m2 = 0
  m3 = 0
  m4 = 1
  tfm_mat = np.array([[m1, m2,tx],[m3, m4, ty]])

  #initial transformation
  #i = 0 to N1
  for i in range(im1_xy.shape[1]):
    try:
      x1[i] = m1*x1[i] + m2*y1[i] + tx 
      y1[i] = m3*x1[i] + m4*y1[i] + ty 
    except:
      ValueError
      OverflowError

  for iter in range(iterations):
  #match_pts is Nx 2 array containing matching points in im2
    match_pts1 = getMatchingPts(x1,y1,x2,y2)

    A_mat = constructA(x1,y1)

    B = constructb(match_pts1)
    X = np.matmul(np.linalg.pinv(A_mat), B)

    m1,m2,m3,m4,tx,ty = X[0],X[1],X[2],X[3],X[4],X[5]
    try:
      for i in range(x1.shape[0]):
        x1[i] = m1*x1[i] + m2*y1[i] + tx 
        y1[i] = m3*x1[i] + m4*y1[i] + ty 
    except:
      ValueError
      OverflowError

  return x1,y1

def evalAlignment(aligned1, im2):
  '''
  Computes the error of the aligned image (aligned1) and im2, as the
  average of the average minimum distance of a point in aligned1 to a point in im2
  and the average minimum distance of a point in im2 to aligned1.
  '''
  d2 = ndimage.distance_transform_edt(1-im2) #distance transform
  err1 = np.mean(np.mean(d2[aligned1 > 0]))
  d1 = ndimage.distance_transform_edt(1-aligned1)
  err2 = np.mean(np.mean(d2[im2 > 0]))
  err = (err1+err2)/2
  return err

def displayAlignment(im1, im2, aligned1, thick=False):
  '''
  Displays the alignment of im1 to im2
     im1: first input image to alignment algorithm (im1(y, x)=1 if (y, x) 
      is an original point in the first image)
     im2: second input image to alignment algorithm
     aligned1: new1(y, x) = 1 iff (y, x) is a rounded transformed point from the first time 
     thick: true if a line should be thickened for display
  ''' 
  if thick:
    # for thick lines (looks better for final display)
    dispim = np.concatenate((cv2.dilate(im1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(aligned1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(im2.astype('uint8'), np.ones((3,3), np.uint8), iterations=1)), axis=-1)
  else:
    # for thin lines (faster)
    dispim = np.concatenate((im1, aligned1, im2), axis = -1)
  return dispim
  
  