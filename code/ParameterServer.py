#!/usr/bin/env python3

import argparse
import socket
import selectors
import time
import errno
import json
from _thread import *

import numpy as np

class ParamMsg():
    def __init__(self, INSTITUTION_ID, MODEL_VERSION, GRADIENT_NUMBER, DATA):
        self.INSTITUTION_ID=INSTITUTION_ID
        self.MODEL_VERSION=MODEL_VERSION
        self.GRADIENT_NUMBER=GRADIENT_NUMBER
        self.DATA=DATA

    def get(self):
        param_msg = [
            self.INSTITUTION_ID,
            self.MODEL_VERSION,
            self.GRADIENT_NUMBER,
            ','.join(list(self.DATA.astype(str)))
        ]
        param_msg = [str(val) for val in param_msg if val is not None]
        param_msg=' '.join(param_msg)+'\n'
        return param_msg.encode()

class ParamServer():
    def __init__(self, PORT):

        self.INSTITUTION_LIST = []

        # I/O Buffer variable
        self.buff_size = int(16*8192)

        # initiation
        self.MODEL_VERSION = 0
        self.KILLSIG = 0
        self.PARAMS = np.random.normal(0, .01, 52650).astype(np.float32)
        self.timer = time.time()
        self.gradients = []
        self.learning_rate=0.01

        self.openServer(PORT)

    def openServer(self, PORT):
        # Open server socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serverSocket:
            serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            serverSocket.bind(('', PORT))
            serverSocket.listen(100)

            try:
                while True:  # Ready to be connected. State maintained unless interrupted.
                    conn, addr = serverSocket.accept()
                    self.INSTITUTION_LIST.append(conn)

                    # prints the address of the user that just connected
                    print(addr[0] + " connected")

                    # creates and individual thread for every client that connects
                    start_new_thread(self.institutionThread, (conn, addr, len(self.INSTITUTION_LIST)))

                    print('.')

            except KeyboardInterrupt:
                self.KILLSIG = 1
                print("Keyboard Interruption. Exiting...")
            finally:
                serverSocket.close()

    def remove(self, connection):
        if connection in self.INSTITUTION_LIST:
            self.INSTITUTION_LIST.remove(connection)

    def institutionThread(self, conn, addr, num=0):
        def read(conn):
            data = b""
            print(f"({num}) receiving...", end=" ")
            while True:
                _recv_buff = conn.recv(int(8192))
                data += _recv_buff
                if len(data) == 0 or data.decode('UTF-8')[-1] == '\n':
                    break
                print('-', end=" ")
            print()
            return data

        while True:
            if self.KILLSIG:
                self.remove(conn)
                break
            try:
                # receive data from the client whose object is conn
                data = read(conn)

                if not data:
                    print(f"{addr} -- No Transfer.")
                    self.remove(conn)
                    break
                else:
                    msg_to_inst = self.process_msgs(data, num)
                    if len(self.gradients) > 0 and time.time() - self.timer > 3:
                        self.PARAMS -= self.learning_rate * np.vstack(self.gradients).mean(0)
                        print(self.PARAMS)
                        self.timer = time.time()
                        self.gradients = []
                        self.MODEL_VERSION += 1
                        print("PARAM UPDATE")

                        msg_to_inst = ParamMsg(
                            INSTITUTION_ID=None,
                            MODEL_VERSION=self.MODEL_VERSION,
                            GRADIENT_NUMBER=None,
                            DATA=self.PARAMS
                        ).get()

                    conn.sendall(msg_to_inst)
                    print("MSG SENT")

            except socket.error as e:
                if e.errno != errno.ECONNRESET:
                    self.remove(conn)
                    raise e
                else:
                    print(f"Client {addr} Disconnected")
                    self.remove(conn)
                    break
            else:
                continue

    def process_msgs(self, data, num=0):
        def parse_msg(data):
            msg = data.decode('UTF-8')
            msg_parsed = msg.strip().split(' ')
            msg_parsed = ParamMsg(
                INSTITUTION_ID=int(msg_parsed[0]),
                MODEL_VERSION=int(msg_parsed[1]),
                GRADIENT_NUMBER=int(msg_parsed[2]),
                DATA=np.array(
                    msg_parsed[3].split(',')
                ).astype(float)
            )
            return msg_parsed

        msg_parsed = parse_msg(data)
        print(f"DATA {msg_parsed.GRADIENT_NUMBER} (MODEL {msg_parsed.MODEL_VERSION}) from INSTITUTION {msg_parsed.INSTITUTION_ID} received. ({num})")
        if msg_parsed.MODEL_VERSION == self.MODEL_VERSION:
            print(f"Model version {self.MODEL_VERSION} matched.")
            self.gradients.append(msg_parsed.DATA)

        msg_to_send = ParamMsg(
            INSTITUTION_ID=None,
            MODEL_VERSION=self.MODEL_VERSION,
            GRADIENT_NUMBER=None,
            DATA=self.PARAMS
        )
        msg_to_send = msg_to_send.get()
        return msg_to_send

with open('paramsvr_params', 'r') as openfile:
    # Reading from json file
    kwparams = json.load(openfile)

ParamServer(**kwparams)
