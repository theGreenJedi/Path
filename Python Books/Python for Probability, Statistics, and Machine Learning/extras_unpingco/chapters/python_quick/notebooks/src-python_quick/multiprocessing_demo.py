# filename multiprocessing_demo.py
import multiprocessing
import time
def worker(k):
   'worker function'
   print 'am starting process %d' % (k)
   time.sleep(10) # wait ten seconds
   print 'am done waiting!'
   return

if __name__ == '__main__':
   for i in range(10):
       p = multiprocessing.Process(target=worker, args=(i,))
       p.start()

