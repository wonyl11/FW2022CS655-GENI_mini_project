#!/usr/bin/env python3

import argparse
import socket
import selectors
import sys
import errno
import pickle
import json

from _thread import *

import numpy as np
import tensorflow as tf

class ClientMsg():
    def __init__(self, CLIENT_ID, DATA_ID, LABEL, DATA):
        self.CLIENT_ID = CLIENT_ID
        self.DATA_ID = DATA_ID
        self.LABEL = LABEL
        self.DATA = DATA

    def get(self):
        client_msg = ' '.join([
            str(self.CLIENT_ID),
            str(self.DATA_ID),
            str(self.LABEL),
            ','.join(list(self.DATA.astype(str)))
        ]) + '\n'
        return client_msg.encode()


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


class Model():
    """Local model assigned to each institute"""
    def __init__(self):
        self.MODEL_VERSION = 0

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='ones', bias_initializer='ones'),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='ones', bias_initializer='ones'),
            tf.keras.layers.Dense(10, kernel_initializer='ones', bias_initializer='ones')
        ])

        self.param_shapes = [[784, 64], [64], [64, 32], [32], [32, 10], [10]]

        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_tape = []

    def inference_only(self, x_input):
        return self.convert_prob_to_label(self.model(x_input))

    def compute_gradient(self, x_input, label_true):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_input)
            loss = self.loss_func(label_true, y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
        label_pred = self.convert_prob_to_label(y_pred)
        self.loss_tape.append(loss.numpy())
        return gradients, label_pred

    def convert_prob_to_label(self, y_pred):
        return np.argmax(y_pred, -1).astype(int)

    def update(self, new_param):
        _sizes = [0] + [np.prod(elem) for elem in self.param_shapes]
        _sizes = np.cumsum(_sizes)
        new_param = [
            new_param[_sizes[i]:_sizes[i + 1]].reshape(self.param_shapes[i])
            for i in range(_sizes.shape[0] - 1)
        ]
        self.model.set_weights(new_param)


def connectParamSvr(host, port):
    '''
    Establish TCP connection. To handle multiplexing, we use selectors module.
    Message I/O and response processing will be handled by ClientState class,
    and keep tracked by selectors.register().
    '''
    paramSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    paramSocket.setblocking(False)
    paramSocket.connect_ex((host, port))
    return paramSocket

class Institution():
    def __init__(self, clientPORT, paramHOST, paramPORT,
                 NUM_INSTITUTIONS, INSTITUTION_ID, max_num=1024):

        self.NUM_INSTITUTIONS = NUM_INSTITUTIONS
        self.INSTITUTION_ID = INSTITUTION_ID
        self.CLIENT_LIST = []

        self.model = Model()

        # I/O Buffer variable
        self.buff_size = int(8192)
        self.msgs_to_send = []

        # initiation
        self.MODEL_VERSION = -1
        self.GRADIENT_NUMBER = 0
        self.KILLSIG = 0

        # measurement
        self.acc_tape = {}

        # connection
        self.paramSOCK = connectParamSvr(paramHOST, paramPORT)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.paramSOCK, selectors.EVENT_WRITE, data=None)

        self.openServer(clientPORT)

    def flush_output(self):
        pickle.dump(
            [self.acc_tape, self.model.loss_tape, self.model.trainable_variables],
            open("output_institution.pkl", 'wb')
        )
        print(self.acc_tape)
        print(self.model.loss_tape)

    def openServer(self, PORT):
        # Open server socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serverSocket:
            serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            serverSocket.bind(('', PORT))
            serverSocket.listen(100)

            try:
                while True:  # Ready to be connected. State maintained unless interrupted.
                    conn, addr = serverSocket.accept()
                    self.CLIENT_LIST.append(conn)

                    # prints the address of the user that just connected
                    print(addr[0] + " connected")

                    # creates and individual thread for every client that connects
                    start_new_thread(self.clientThread, (conn, addr, len(self.CLIENT_LIST)))

            except KeyboardInterrupt:
                self.KILLSIG = 1
                print("Keyboard Interruption. Exiting...")
                print(len(self.model.loss_tape))
            finally:
                serverSocket.close()

    def remove(self, connection):
        if connection in self.CLIENT_LIST:
            self.CLIENT_LIST.remove(connection)

    def clientThread(self, connClient, addrClient, num=0):
        def read(conn, num):
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

        def writePARAM():
            if self._send_buff:
                try:
                    sent = self.paramSOCK.send(self._send_buff)
                    self._sent_buff += self._send_buff[:sent]
                except BlockingIOError:
                    pass
                else:
                    self._send_buff = self._send_buff[sent:]

        def paramsvr_connection(key, mask):
            sock = key.fileobj

            if mask & selectors.EVENT_READ:
                msg_from_param = read(sock, "PARAM")  # Should be ready to read
                msg_from_param = msg_from_param.decode('UTF-8').strip().split(' ')
                msg_from_param[0] = int(msg_from_param[0])
                msg_from_param[1] = np.array(
                    msg_from_param[1].split(',')
                ).astype(float)
                if msg_from_param[0] > self.MODEL_VERSION:
                    print(msg_from_param[1])
                    self.model.update(msg_from_param[1])
                    self.MODEL_VERSION += 1
                    print("MODEL UPDATED")
                self.sel.modify(sock, selectors.EVENT_WRITE, data=self)
                return 1

            if mask & selectors.EVENT_WRITE:
                writePARAM()
                self.sel.modify(sock, selectors.EVENT_READ, data=self)
                return 0

        while True:
            if self.KILLSIG:
                self.remove(connClient)
                break
            try:
                # receive data from the client whose object is conn
                data = read(connClient, num)

                if not data:
                    print(f"{addrClient} -- No Transfer.")
                    self.remove(connClient)
                    if len(self.CLIENT_LIST) == 0:
                        self.flush_output()
                    return 0
                else:
                    msg_to_client, isCorrect = self.process_client_msgs(data, num)
                    if self.MODEL_VERSION in self.acc_tape.keys():
                        self.acc_tape[self.MODEL_VERSION].append(isCorrect)
                    else:
                        self.acc_tape[self.MODEL_VERSION] = [isCorrect]

                    # send prediction to the client
                    connClient.sendall(msg_to_client)

                if self.msgs_to_send:
                    self._sent_buff = b""
                    self._send_buff = self.msgs_to_send[0]
                    self.msgs_to_send = self.msgs_to_send[1:]
                    print("sending to param svr...")
                    _ = 0
                    while True:
                        events = self.sel.select(timeout=1)
                        if events:
                            for key, mask in events:
                                _ = paramsvr_connection(key, mask)

                        # Check for a socket being monitored to continue.
                        if not self.sel.get_map() or _ == 1:
                            break

            except socket.error as e:
                if e.errno != errno.ECONNRESET:
                    self.remove(connClient)
                    raise e
                else:
                    print(f"Client {addrClient} Disconnected")
                    self.remove(connClient)
                    break
            else:
                continue

    def process_client_msgs(self, data, num=0):
        def parse_client_msg(data):
            client_msg = data.decode('UTF-8')
            client_msg_parsed = client_msg.strip().split(' ')
            client_msg_parsed = ClientMsg(
                CLIENT_ID=int(client_msg_parsed[0]),
                DATA_ID=int(client_msg_parsed[1]),
                LABEL=int(client_msg_parsed[2]),
                DATA=np.array(
                    client_msg_parsed[3].split(',')
                ).astype(int)
            )
            return client_msg_parsed

        def compute_gradient(client_msg_parsed):
            gradient_number = self.GRADIENT_NUMBER
            gradients, label_pred = self.model.compute_gradient(
                x_input=client_msg_parsed.DATA[np.newaxis, :],
                label_true=client_msg_parsed.LABEL
            )
            self.GRADIENT_NUMBER += 1
            gradients = np.concatenate([val.numpy().flatten() for val in gradients])
            return gradients, gradient_number, label_pred[0]

        client_msg_parsed = parse_client_msg(data)
        print(f"DATA {client_msg_parsed.DATA_ID} from CLIENT {client_msg_parsed.CLIENT_ID} received. ({num})")

        gradients, gradient_number, label_pred = compute_gradient(client_msg_parsed)

        client_msg_parsed.DATA = label_pred
        msg_to_client = client_msg_parsed.get()

        msg_to_server = ParamMsg(
            INSTITUTION_ID=self.INSTITUTION_ID,
            MODEL_VERSION=self.MODEL_VERSION,
            GRADIENT_NUMBER=self.GRADIENT_NUMBER,
            DATA=gradients
        )
        msg_to_server = msg_to_server.get()
        self.msgs_to_send.append(msg_to_server)
        isCorrect = label_pred == client_msg_parsed.LABEL
        print(f"True: {client_msg_parsed.LABEL} PRED: {label_pred} isCorrect: {isCorrect}")
        return msg_to_client, isCorrect

with open('institution_params', 'r') as openfile:
    # Reading from json file
    kwparams = json.load(openfile)

Institution(**kwparams)

