# filename: concurrent_demo.py
import futures
import time

def worker(k):
   'worker function'
   print 'am starting process %d' % (k)
   time.sleep(10) # wait ten seconds
   print 'am done waiting!'
   return

def main():
   with futures.ProcessPoolExecutor(max_workers=3) as executor:
      list(executor.map(worker,range(10)))

if __name__ == '__main__':
    main()
