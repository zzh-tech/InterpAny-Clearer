import redis
from pprint import pprint
import time

rcli = redis.Redis()

pprint("Idle")
rcli.set('dancetype', 0)
time.sleep(10)

rcli.set('dancetype', 3)
rcli.set('tempo', 120)
time.sleep(20)

pprint("Idle")
rcli.set('dancetype', 0)
