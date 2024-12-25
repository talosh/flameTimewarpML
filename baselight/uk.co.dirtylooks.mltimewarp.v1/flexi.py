# Flexi Python module
#
# Copyright (c) 2021 FilmLight. All rights reserved.
# npublished - rights reserved under the copyright laws of the
# United States. Use of a copyright notice is precautionary only
# and does not imply publication or disclosure.
# This software contains confidential information and trade secrets
# of FilmLight Limited. Use, disclosure, or reproduction
# is prohibited without the prior express written permission of
# FilmLight Limited.

import sys
import json
import websocket
import traceback
import os

# Globals
Flexi_registered_effects = {}
Flexi_effect_for_id = {}
Flexi_ws = None

def register_id(id, effect):
    global Flexi_effect_for_id, Flexi_registered_effects
    Flexi_registered_effects[effect] = 1
    Flexi_effect_for_id[id] = effect

def run():
    # Open connection to the application using the URL passed in the environment
    global Flexi_ws

    if(not 'FLEXI_SERVER_URL' in os.environ):
        print('flexi.run() called without websocket URL. Running in test mode.')
        from jsonschema import validate
        with open('flexi_reply.schema') as f:
            describe_effect_schema = json.load(f)
        class DummyWS:
            def send(self, s):
                self.reply = json.loads(s)
#                print(json.dumps(self.reply, indent=2))
#                if 'log' in self.reply:
#                    print(json.dumps(reply['log'], indent=2))
#                if 'data' in self.reply:
#                    print(json.dumps(reply['data'], indent=2))
        Flexi_ws = DummyWS()
        for id in Flexi_effect_for_id:
            print(f'Validating describe_effect for {id[0]} v{id[1]}')
            _process_message(
                {'effect' : [id[0], id[1], 'instance'],
                 'method' : 'describe_effect_v1',
                 'id' : 2,
                 'data' : {}})
            validate(Flexi_ws.reply, describe_effect_schema)
        return

    url = os.environ['FLEXI_SERVER_URL']
       
    Flexi_ws = websocket.create_connection(url)

#    print('Connected to {}'.format(url))

    # Processing loop
    while True:
        try:
            message = json.loads(Flexi_ws.recv())
        except websocket._exceptions.WebSocketConnectionClosedException:
            print('Connection closed')
            _shutdown()
        if isinstance(message, list):
            # TODO could consider batching replies
            for entry in message:
                _process_message(entry)
        else:
            _process_message(message)

# Process one message from the application
def _process_message(message):
    global Flexi_effect_for_id, Flexi_registered_effects

    if 'method' in message and message['method'] == 'shutdown_v1':
        _reply(message, {})
        _shutdown()

    try:
        id = (message['effect'][0], message['effect'][1])
        if not id in Flexi_effect_for_id:
            return _reply(message, error('Unregistered effect', id[0]))
        effect = Flexi_effect_for_id[id]
        effect._handle_message(id, message)
        
    except SystemExit as e:
        sys.exit(e)
        
    except:
        print('Internal error', format(sys.exc_info()))
        traceback.print_exc()
        _reply(message, error('Internal error', format(sys.exc_info())))

# Send a reply to a message
def _reply(message, data):
    global Flexi_ws
    if data != None and '__LOG__' in data:
        reply = {'effect' : message['effect'],
                 'method' : message['method'].upper(),
                 'id' : message['id'],
                 'log' : data['__LOG__']
                }
    else:
        reply = {'effect' : message['effect'],
                 'method' : message['method'].upper(),
                 'id' : message['id'],
                 'data' : data
                 }
    Flexi_ws.send(json.dumps(reply))

# Make a reply with an error to a message
def error(error, detail):
    return { '__LOG__' : [ { 'level' : 3, 'message' : error, 'detail' : detail } ] }

def _shutdown():
    global Flexi_ws
    for id in Flexi_effect_for_id:
        effect = Flexi_effect_for_id[id]
        effect.shutdown()
    if Flexi_ws != None:
        Flexi_ws.close()
    print('Exiting')
    sys.exit(0)

# Effect class - subclass this to define your effect

class Effect:

    def init(self, ids):
        """Initialise the effect object.
        ids is a list of tuples ('identifier', version)
        """
        for id in ids:
            register_id(id, self)

    def _handle_message(self, id, message):
        """Internal method"""
        gpu = None
        if 'data' in message and 'cuda_gpu' in message['data']:
            cuda_gpu = message['data']['cuda_gpu']
            if cuda_gpu != '':
                gpu = cuda_gpu
        if 'data' in message and 'metal_device_index' in message['data']:
            metal_device_index = message['data']['metal_device_index']
            if metal_device_index != -1:
                gpu = metal_device_index
        if message['method'] == 'describe_effect_v1':
            _reply(message, self.describe_effect(id))
            self.setup_if_necessary(id, gpu)
        elif message['method'] == 'query_vram_requirement_v1':
            _reply(message, self.query_vram_requirement(id, message['data']))
        elif message['method'] == 'set_vram_limit_v1':
            res = self.set_vram_limit(id, message['data'])
            if(res == 'quit'):
                _reply(message, {})
                _shutdown()
            else:
                _reply(message, res)
        elif message['method'] == 'run_generate_v1':
            self.setup_if_necessary(id, gpu)
            _reply(message, self.run_generate(id, message['effect'][2], message['data']))
        else:
            _reply(message, error('Unknown method', message['method']))
            
    def describe_effect(self, id, message):
        """Subclass must override this to QUICKLY return a description of a particular effect.
        id is a tuple ('identifier', version, 'instance')
        message will have members 'host_identifer', 'host_version' which can be used
        to change how the effect behaves, or indeed to fail to describe it
        """
        raise RuntimeError('Effect subclass must implement describe_effect() method')

    def setup_if_necessary(self, id, gpu):
        """Subclass may override this to do expensive setup.
        gpu is (on Linux) a string which is the CUDA UUID of the GPU to use,
        or (on macOS) an integer index into the result of MTLCopyAllDevices()"""
        return

    def query_vram_requirement(self, id, data):
        """Subclass must override this to QUICKLY return the VRAM requirement of a particular effect.
        id is a tuple ('identifier', version, 'instance')
        data contains params, width and height
        Reply must contain 'min_mb' and 'max_mb'
        """
        raise RuntimeError('Effect subclass must implement query_vram_requirement() method')

    def set_vram_limit(self, id, data):
        """Subclass must override this to IMMEDIATELY restrict its current and future
        VRAM usage to the supplied value data['limit_mb'] in MB.
        
        If limit_mb is zero, the effect must release ALL resources on the indicated device.
        
        Method must not return until its GPU usage is at or below the supplied limit,
        except in the special case where a GPU SDK is being used which does not support
        releasing resources, then this method can return 'exit', which will cause the
        process to exit.
        
        id is a tuple ('identifier', version, 'instance')
        """
        raise RuntimeError('Effect subclass must implement set_vram_limit() method')

    def run_generate(self, id, instance, data):
        """Subclass must override this to run an effect.
        id is a tuple ('identifier', version, 'instance')
        instance is a string that might remain constant for each instance of the effect
        in the app
        data contains the inputs, outputs, parameters etc
        """
        raise RuntimeError('Effect subclass must implement run_generate() method')

    def shutdown(self):
        """Subclass may override this to do a cleanup prior to shutdown"""
        return

# Exports
__all__ = ["Effect", "run"]
