import socket
import cv2
import numpy as np
import struct

class RovCam():
    """The camera module responsible for all camera connections to ROV, based on reusable sockets and openCV."""

    # max length of a single packet (which is more than enough to send over a full frame)
    MAX_DATAGRAM = 1450
    FRONT = 0
    ARM = 1
    MISC1 = 2
    MISC2 = 3

    def __init__(self, port_num):
        self.port = port_num
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # self.s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10 * self.MAX_DATAGRAM)

        self.s.bind(("", self.port))
        print("ROV CAM : Connected successfully")

    def __str__(self):
        return f"ROV Camera instance connected on rov camera {self.__cam}"
    def read(self):
        # handle errors
        frame_details = 0
        while frame_details == 0:
          try:
            frame_details = self.s.recvfrom(3 * 32)
          except Exception as e:
            print(e)
        my_bytes = frame_details[0]
        uint32_array = []
        # uint32_array = [struct.unpack('<I', my_bytes[i:i+4])[0] for i in range(0, len(my_bytes), 4)]

        uint32_array = []
        for i in range(0, len(my_bytes), 4):
          uint32_array.append(struct.unpack('<I', my_bytes[i:i+4])[0])

        #print (uint32_array)
        packetSize, imageSize, _ = uint32_array
    
        imageBuffer = b''
        for i in range(packetSize):
          imageBuffer += self.s.recvfrom(self.MAX_DATAGRAM)[0]
        # print(imageBuffer)
        imageNumpy = np.frombuffer(imageBuffer, dtype=np.uint8)
        # data = pickle.loads(data)
        frame = cv2.imdecode(imageNumpy, cv2.IMREAD_COLOR)
        return frame
 


if __name__=='__main__':
    cam = RovCam(5800)
    while True:   
      try:
        frame = cam.read()
        cv2.imshow("Main Receiving", frame)
        cv2.waitKey(1)
      except Exception as e:
        #print(e)
        pass
      