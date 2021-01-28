
import struct

fp = open('test.jbig2', 'rb')

# headerFlags, = struct.unpack('>B', fp.read(1))

# fileOrganisation = headerFlags & 1
# randomAccessOrganisation = fileOrganisation == 0

# pagesKnown = headerFlags & 2
# noOfPagesKnown = pagesKnown == 0
# print('headerFlags:', headerFlags)

segmentNumber = struct.unpack('>4H', fp.read(8))
segmentHeaderFlags, = struct.unpack('>B', fp.read(1))
# referedToSegmentCountAndRetentionFlags, = struct.unpack('>B', fp.read(1))
# referredToSegmentCount = (referedToSegmentCountAndRetentionFlags & 224) >> 5
# retentionFlags = (referedToSegmentCountAndRetentionFlags & 31)

print('segmentNumber:', segmentNumber)
print('segmentHeaderFlags:', segmentHeaderFlags)
# print('referredToSegmentCount:', referredToSegmentCount)
# print('retentionFlags:', retentionFlags)

fp.close()