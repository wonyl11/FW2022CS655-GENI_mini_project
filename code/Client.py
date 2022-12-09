#!/usr/bin/env python3

import socket
import pickle
import selectors
import time
import json
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


class Client():
    def __init__(self, sel, sock,
                 NUM_INSTITUTIONS, INSTITUTION_ID, CLIENT_ID,
                 data_imbalance_opt=False, max_num=1024, timewait_mean=.0001):

        self.NUM_INSTITUTIONS = NUM_INSTITUTIONS
        self.INSTITUTION_ID = INSTITUTION_ID
        self.CLIENT_ID = CLIENT_ID

        # Config for experiment
        self.data_imbalance_opt = data_imbalance_opt
        self.max_num = max_num
        self.timewait_mean = timewait_mean

        # Selector and Socket
        self.sel = sel
        self.sock = sock

        # I/O Buffer and state chk variables
        self.buff_size = int(8192)
        self._send_buff = b""
        self._sent_buff = b""
        self._recv_buff = b""
        self.sending = False

        # for measurements
        self.time_send_start = []
        self.time_send_end = []
        self.time_recv_start = []
        self.time_recv_end = []
        self.rtt_measurements = []
        self.tput_measurements = []

        # initiation
        self.train_images, self.train_labels = self.init_dataset()
        self.ClientMsg = ClientMsg(
            CLIENT_ID=self.CLIENT_ID,
            DATA_ID=0,
            LABEL=None,
            DATA=None
        )

    def init_dataset(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (_, _) = fashion_mnist.load_data()

        _datainds = np.array_split(
            np.arange(train_images.shape[0]),
            self.NUM_INSTITUTIONS
        )[self.INSTITUTION_ID]

        if self.data_imbalance_opt:
            num_labels = np.unique(train_labels).shape[0]
            _my_labels = np.array_split(
                np.arange(num_labels),
                self.NUM_INSTITUTIONS
            )[self.INSTITUTION_ID]
            _datainds = _datainds[np.isin(train_labels[_datainds], _my_labels)]

        np.random.shuffle(_datainds)
        _datainds = _datainds[:self.max_num]

        train_images = train_images[_datainds]
        train_images = train_images.reshape(train_images.shape[0], np.prod(train_images.shape[1:]))
        train_labels = train_labels[_datainds]
        return train_images, train_labels

    def generate_msg(self):
        self.ClientMsg.DATA = self.train_images[self.ClientMsg.DATA_ID]
        self.ClientMsg.LABEL = self.train_labels[self.ClientMsg.DATA_ID]
        return self.ClientMsg.get()

    def set_selector_mode(self, mode):
        """Set selector event to either read ('r') or write ('w')"""
        if mode == "r":
            events = selectors.EVENT_READ
        elif mode == "w":
            events = selectors.EVENT_WRITE
        else:
            raise Exception(f"Incorrect mode {mode}")
        self.sel.modify(self.sock, events, data=self)

    def process(self, mask):
        if mask & selectors.EVENT_READ:
            self.read()
            if self._recv_buff.decode('UTF-8')[-1] == '\n':
                self.time_recv_end.append(time.time())
                self.proceed()

        if mask & selectors.EVENT_WRITE:
            self.write()

    def read(self):
        print("receiving...\n")
        try:
            if not self._recv_buff:
                self.time_recv_start.append(time.time())
            data = self.sock.recv(self.buff_size)
        except BlockingIOError:
            pass
        else:
            if data:
                self._recv_buff += data
            else:
                raise Exception("disconnected.")

    def proceed(self):
        self.rtt_measurements.append(self.time_recv_end[-1] - self.time_send_start[-1])
        self.tput_measurements.append(
            np.divide(self.MESSAGE_SIZE, (self.time_recv_end[-1] - self.time_send_start[-1]))
        )

        response = self._recv_buff.decode("UTF-8")
        print(response[:128],'...\n')
        if self.ClientMsg.DATA_ID < self.max_num - 1:
            self.set_selector_mode("w")
            self.ClientMsg.DATA_ID += 1
        else:
            self.close()
            return 0

        # simulation for input waiting time interval
        if self.timewait_mean > 0:
            _t = np.random.poisson(self.timewait_mean)
            time.sleep(_t)

        self._recv_buff = b""
        self.write()

    def write(self):
        def _write():
            if self._send_buff:
                try:
                    self.time_send_start.append(time.time())
                    sent = self.sock.send(self._send_buff)
                    self._sent_buff += self._send_buff[:sent]
                except BlockingIOError:
                    pass
                else:
                    self._send_buff = self._send_buff[sent:]

        print("sending...\n")
        if not self.sending:
            self._send_buff = self.generate_msg()
            self.MESSAGE_SIZE = len(self._send_buff)
            self.sending = True

        _write()

        if self.sending:
            if not self._send_buff:
                self.time_send_end.append(time.time())
                print(f"DATA ID: {self.ClientMsg.DATA_ID}\n")
                print(f"msg sent: {self._sent_buff[:128]}...\n")
                self._sent_buff = b""

                # Set the client to read mode.
                self.set_selector_mode("r")
                self.sending = False

    def close(self):
        try:
            self.sel.unregister(self.sock)
        except Exception as e:
            print(
                f"Error: selector.unregister() {e!r}"
            )
        try:
            self.sock.close()
        except OSError as e:
            print(f"Error: socket.close() {e!r}")
        finally:
            self.sock = None


def connectClient(host, port, sel, kwparams):
    '''
    Establish TCP connection. To handle multiplexing, we use selectors module.
    Message I/O and response processing will be handled by ClientState class,
    and keep tracked by selectors.register().
    '''
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientSocket.setblocking(False)
    clientSocket.connect_ex((host, port))

    # Init Client
    state = Client(sel, clientSocket, **kwparams)

    # Init. selector with writing mode
    sel.register(clientSocket, selectors.EVENT_WRITE, data=state)


def executeClient(HOST, PORT, kwparams):
    sel = selectors.DefaultSelector()
    connectClient(HOST, PORT, sel, kwparams)

    try:
        while True:
            events = sel.select()
            if events:
                for key, mask in events:
                    client_state = key.data
                    client_state.process(mask)
                    # try:
                    #     client_state.process(mask)
                    # except Exception as e:
                    #     print("While Executing, we found: ", e)
                    #     client_state.close()
            if not sel.get_map():
                break

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")
    finally:
        sel.close()

    output = [
        client_state.time_send_start,
        client_state.time_send_end,
        client_state.time_recv_start,
        client_state.time_recv_end,
        client_state.rtt_measurements,
        client_state.tput_measurements
    ]
    return output

with open('client_params', 'r') as openfile:
    # Reading from json file
    kwparams = json.load(openfile)

HOST = kwparams["HOST"]
PORT = kwparams["PORT"]

kwparams = {
    "NUM_INSTITUTIONS": kwparams["NUM_INSTITUTIONS"],
    "INSTITUTION_ID": kwparams["INSTITUTION_ID"],
    "CLIENT_ID": kwparams["CLIENT_ID"]
}

output = executeClient(HOST, PORT, kwparams)
pickle.dump(output, open("output_client.pkl", "wb"))


