import re

#
# this re will match a line of the form:
#
# 'Wheelbase: 98" 	Track: 58.7" Front / 59.4" Rear 	Height: 47.8" Coupe'
#
# and will assign the wheelbase number (no quote), track front, track rear,
# and height to groups
#
WHEELB_GROUP = 0
TRACKF_GROUP = 0
TRACKR_GROUP = 0
HEIGHT_GROUP = 0
dataPatt = re.compile("$")

#
# run test if this module is run as a script
#
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:

        fileHandle = open(sys.argv[1], "r")
        for line in fileHandle.readlines():
            dataMatch = dataPatt.match(line)
            if dataMatch:
                print "WHEELBASE: %s" % dataMatch.group(WHEELB_GROUP)
                print "FRONT TRACK: %s" % dataMatch.group(TRACKF_GROUP)
                print "REAR TRACK: %s" % dataMatch.group(TRACKR_GROUP)
                print "HEIGHT: %s" % dataMatch.group(HEIGHT_GROUP)

    else:
        line = 'Wheelbase: 98" 	Track: 58.7" Front / 59.4" Rear 	Height: 47.8" Coupe'
        dataMatch = dataPatt.match(line)
        if dataMatch:
            print "WHEELBASE: %s" % dataMatch.group(WHEELB_GROUP)
            print "FRONT TRACK: %s" % dataMatch.group(TRACKF_GROUP)
            print "REAR TRACK: %s" % dataMatch.group(TRACKR_GROUP)
            print "HEIGHT: %s" % dataMatch.group(HEIGHT_GROUP)

        else:
            print 'RE did not match!'
