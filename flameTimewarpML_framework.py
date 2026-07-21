import os
import sys
import time
import queue
import threading
import traceback
import atexit
import hashlib
import pickle

from pprint import pprint, pformat

class flameAppFramework(object):
    # flameAppFramework class takes care of preferences and more stuff

    class prefs_dict(dict):
        # subclass of a dict() in order to directly link it 
        # to main framework prefs dictionaries
        # when accessed directly it will operate on a dictionary under a 'name'
        # key in master dictionary.
        # master = {}
        # p = prefs(master, 'app_name')
        # p['key'] = 'value'
        # master - {'app_name': {'key', 'value'}}
            
        def __init__(self, master, name, **kwargs):
            self.name = name
            self.master = master
            if not self.master.get(self.name):
                self.master[self.name] = {}
            self.master[self.name].__init__()

        def __getitem__(self, k):
            return self.master[self.name].__getitem__(k)
        
        def __setitem__(self, k, v):
            return self.master[self.name].__setitem__(k, v)

        def __delitem__(self, k):
            return self.master[self.name].__delitem__(k)
        
        def get(self, k, default=None):
            return self.master[self.name].get(k, default)
        
        def setdefault(self, k, default=None):
            return self.master[self.name].setdefault(k, default)

        def pop(self, k, v=object()):
            if v is object():
                return self.master[self.name].pop(k)
            return self.master[self.name].pop(k, v)
        
        def update(self, mapping=(), **kwargs):
            self.master[self.name].update(mapping, **kwargs)
        
        def __contains__(self, k):
            return self.master[self.name].__contains__(k)

        def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
            return type(self)(self)
        
        def keys(self):
            return list(self.master[self.name].keys())

        @classmethod
        def fromkeys(cls, keys, v=None):
            return self.master[self.name].fromkeys(keys, v)
        
        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, self.master[self.name].__repr__())

        def master_keys(self):
            return list(self.master.keys())
    
    class MinExrReader:
        '''Minimal, standalone OpenEXR reader for single-part, uncompressed scan line files.

        This OpenEXR reader makes a couple of assumptions
        - single-part files with arbitrary number of channels,
        - no pixel data compression, and
        - equal channel types (HALF, FLOAT, UINT).

        These assumptions allow us to efficiently parse and read the `.exr` file. In particular
        we gain constant offsets between scan lines which allows us to read the entire image
        in (H,C,W) format without copying.

        Use `MinimalEXR.select` to select a subset of channels in the given order. `MinimalEXR.select`
        tries to be smart when copying is required and when views are ok.
        
        Based on the file format presented in
        https://www.openexr.com/documentation/openexrfilelayout.pdf

        Attributes
        ----------
        shape: tuple
            Shape of image in (H,C,W) order
        image: numpy.array
            Uncompressed image data.
        attrs: dict
            OpenEXR header attributes.
        '''

        class BufferReader:
            '''A lightweight io.BytesIO object with convenience functions.
            
            Params
            ------
            data : bytes-like
                Bytes for which random access is required.
            
            '''

            def __init__(self, data):
                self.data = data
                self.len = len(data)
                self.off = 0

            def read(self, n):
                '''Read next `n` bytes.'''
                v = self.data[self.off:self.off+n]
                self.off += n
                return v

            def read_null_string(self):
                import ctypes
                '''Read a null-terminated string.'''
                s = ctypes.create_string_buffer(self.data[self.off:]).value
                if s != None:
                    s = s.decode('utf-8')
                    self.off += len(s) + 1
                return s

            def peek(self):
                '''Peek next byte.'''
                return self.data[self.off]

            def advance(self, n):
                '''Advance offset by `n` bytes.'''
                self.off += n

            def nleft(self):
                '''Returns the number of bytes left to read.'''
                return self.len - self.off - 1

        def __init__(self, fp, header_only = False):
            self.fp = fp
            self.image = None
            self.shape = None

            self._read_header()
            if not header_only:
                self._read_image()

        def select(self, channels, channels_last=True):
            import numpy as np
            '''Returns an image composed only of the given channels.
            
            Attempts to be smart about memory views vs. memory copies.

            Params
            ------
            channels: list-like
                Names of channels to be extracted. Appearance in list
                also defines the order of the channels. 
            channels_last: bool, optional
                When true return image in (H,W,C) format.

            Returns
            -------
            image: HxWxC or HxCxW array
                Selected image data.
            '''
            H,C,W = self.shape
            ids = [self.channel_map[c] for c in channels]                
            if len(ids) == 0:
                img = np.empty((H,0,W), dtype=self.image.dtype)
            else:
                diff = np.diff(ids)
                sH = slice(0, H)
                sW = slice(0, W)
                if len(diff) == 0:
                    # single channel select, return view
                    sC = slice(ids[0],ids[0]+1)
                    img = self.image[sH,sC,sW]
                elif len(set(diff)) == 1:
                    # mutliple channels, constant offset between, return view
                    # Careful here with negative steps, ie. diff[0] < 0:
                    start = ids[0]
                    step = diff[0]
                    end = ids[-1]+diff[0]
                    end = None if end < 0 else end                
                    sC = slice(start,end,step)
                    img = self.image[sH,sC,sW]
                else:
                    # multiple channels not slicable -> copy mem
                    chdata = [self.image[sH,i:i+1,sW] for i in ids]
                    img = np.concatenate(chdata, 1)
            
            if channels_last:
                img = img.transpose(0,2,1)
            return img

        def _read_header(self):
            import numpy as np
            import struct

            self.fp.seek(0)        
            buf = self.BufferReader(self.fp.read(10000))

            # Magic and version and info bits
            magic, version, b2, b3, b4 = struct.unpack('<iB3B', buf.read(8))
            assert magic == 20000630, 'Not an OpenEXR file.'
            assert b2 in (0, 4), 'Not a single-part scan line file.'
            assert b3 == b4 == 0, 'Unused flags in version field are not zero.'

            # Header attributes
            self.attrs = self._read_header_attrs(buf)

            # Parse channels and datawindow
            self.compr = self._parse_compression(self.attrs)        
            self.channel_names, self.channel_types = self._parse_channels(self.attrs)
            self.channel_map = {cn:i for i,cn in enumerate(self.channel_names)}
            H, W = self._parse_data_window(self.attrs)
            self.shape = (H,len(self.channel_names),W)
            self.first_offset = self._read_first_offset(buf)
            
            # Assert our assumptions
            assert self.compr == 0x00, 'Compression not supported.'
            assert len(set(self.channel_types)) <= 1, 'All channel types must be equal.'

        def _read_image(self):
            import numpy as np
            # Here is a shortcut: We assume all channels of the same type and thus constant offsets between
            # scanlines (SOFF). Note, each scanline has a header (y-coordinate (int4), data size DS (int4)) and data in scanlines
            # is stored consecutively for channels (in order of appearance in header). Thus we can interpret the content
            # as HxCxW image with strides: (SOFF,DS*W,DS)
            H,C,W = self.shape

            if np.prod(self.shape) == 0:
                return np.empty(self.shape, dtype=np.float32)

            dtype  = self.channel_types[0]
            DS = np.dtype(dtype).itemsize
            SOFF = 8+DS*W*C        
            strides = (SOFF, DS*W, DS)
            nbytes = SOFF*H

            self.fp.seek(self.first_offset, 0)
            image = np.frombuffer(self.fp.read(nbytes), dtype=dtype, count=-1, offset=8)
            self.image = np.lib.stride_tricks.as_strided(image, (H,C,W), strides)

        def _read_header_attrs(self, buf):
            attrs = {}
            while buf.nleft() > 0:
                attr = self._read_header_attr(buf)
                if attr is None:
                    break
                attrs[attr[0]] = attr
            return attrs

        def _read_header_attr(self, buf):
            import struct
            if buf.peek() == 0x00:
                buf.advance(1)
                return None
            aname = buf.read_null_string()
            atype = buf.read_null_string()
            asize = struct.unpack('<i', buf.read(4))[0]
            data = buf.read(asize)
            return (aname, atype, asize, data)

        def _parse_channels(self, attrs):
            import struct
            import numpy as np

            attr = attrs['channels']
            assert attr[1] == 'chlist', 'Unexcepted type for channels attribute.'
            buf = self.BufferReader(attr[-1])
            channel_names, channel_types = [], []
            PT_LOOKUP = [np.uint32, np.float16, np.float32]
            while buf.nleft() > 0 and buf.peek() != 0x00:            
                channel_names.append(buf.read_null_string())
                pt = struct.unpack('<i', buf.read(4))[0]
                channel_types.append(PT_LOOKUP[pt])
                buf.advance(12) # skip remaining entries
            if buf.nleft() > 0:
                buf.advance(1) # account for zero byte
            return channel_names, channel_types

        def _parse_data_window(self, attrs):
            import struct
            attr = attrs['dataWindow']
            assert attr[1] == 'box2i', 'Unexcepted type for dataWindow attribute.'
            xmin, ymin, xmax, ymax = struct.unpack('<iiii', attr[-1])
            return (ymax-ymin+1, xmax-xmin+1)

        def _parse_compression(self, attrs):
            return attrs['compression'][-1][0]

        def _read_offsets(self, buf):
            import struct
            offsets = []
            while buf.nleft() > 0 and buf.peek() != 0x00:
                o = struct.unpack('<Q', buf.read(8))[0]
                offsets.append(o)
            if buf.nleft() > 0:
                buf.advance(1) # account for zero byte
            return offsets

        def _read_first_offset(self, buf):
            import struct
            assert buf.nleft() > 0 and buf.peek() != 0x00, 'Failed to read offset.'
            return struct.unpack('<Q', buf.read(8))[0]

    def __init__(self, *args, **kwargs):
        self.name = self.__class__.__name__
        self.settings = kwargs.get('settings', dict())
        self.app_name = self.settings.get('app_name', 'flameApp')
        self.bundle_name = self.sanitize_name(self.app_name)
        self.version = self.settings.get('version', 'Unknown version')
        self.debug = self.settings.get('debug', False)
        self.requirements = self.settings.get('requirements', list())

        self.log_debug(f'settings: {self.settings}')

        # self.prefs scope is limited to flame project and user
        self.prefs = {}
        self.prefs_user = {}
        self.prefs_global = {}
        
        try:
            import flame
            self.flame = flame
            self.flame_project_name = self.flame.project.current_project.name
            self.flame_user_name = flame.users.current_user.name
        except:
            self.flame = None
            self.flame_project_name = 'UnknownFlameProject'
            self.flame_user_name = 'UnknownFlameUser'
        
        try:
            import socket
            self.hostname = socket.gethostname()
        except:
            self.hostname = 'UnknownHostName'

        if self.settings.get('prefs_folder'):
            self.prefs_folder = self.settings['prefs_folder']        
        elif sys.platform == 'darwin':
            self.prefs_folder = os.path.join(
                os.path.expanduser('~'),
                 'Library',
                 'Preferences',
                 self.bundle_name)
        elif sys.platform.startswith('linux'):
            self.prefs_folder = os.path.join(
                os.path.expanduser('~'),
                '.config',
                self.bundle_name)

        self.prefs_folder = os.path.join(
            self.prefs_folder,
            self.hostname,
        )

        self.log_debug('[%s] waking up' % self.__class__.__name__)
        self.load_prefs()
        
        if self.settings.get('bundle_folder'):
            self.bundle_folder = self.settings['bundle_folder']
        else:
            self.bundle_folder = os.path.realpath(
                os.path.dirname(__file__)
            )

        if self.settings.get('packages_folder'):
            self.packages_folder = self.settings['packages_folder']
        else:
            self.packages_folder = os.path.join(
                self.bundle_folder,
                'packages'
            )

        self.site_packages_folder = os.path.join(
            self.packages_folder,
            '.lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages'
        )

        self.log_debug(f'site-packages folder: {self.site_packages_folder}')

        if self.settings.get('temp_folder'):
            self.temp_folder = self.settings['temp_folder']
        else:
            self.temp_folder = os.path.join(
            '/var/tmp',
            self.bundle_name,
            'temp'
        )
        
        self.log_debug(f'temp folder: {self.temp_folder}')

        '''
        self.bundle_path = os.path.join(
            self.bundle_folder,
            self.bundle_name
        )

        if not self.check_bundle_id():
            threading.Thread(
                target=self.unpack_bundle,
                args=(os.path.dirname(self.site_packages_folder), )
            ).start()
        '''

    def log(self, message):
        try:
            print ('[%s] %s' % (self.bundle_name, str(message)))
        except:
            pass

    def log_debug(self, message):
        if self.debug:
            try:
                print ('[DEBUG %s] %s' % (self.bundle_name, str(message)))
            except:
                pass

    def load_prefs(self):
        import json
        
        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs.json'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs.json'
        prefs_global_file_path = prefix + '.prefs.json'

        try:
            with open(prefs_file_path, 'r') as prefs_file:
                self.prefs = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs, indent=4))
        except Exception as e:
            self.log_debug('unable to load preferences from %s' % prefs_file_path)
            self.log_debug(e)

        try:
            with open(prefs_user_file_path, 'r') as prefs_file:
                self.prefs_user = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_user_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_user, indent=4))
        except Exception as e:
            self.log_debug('unable to load preferences from %s' % prefs_user_file_path)
            self.log_debug(e)

        try:
            with open(prefs_global_file_path, 'r') as prefs_file:
                self.prefs_global = json.load(prefs_file)
            self.log_debug('preferences loaded from %s' % prefs_global_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_global, indent=4))
        except Exception as e:
            self.log_debug('unable to load preferences from %s' % prefs_global_file_path)
            self.log_debug(e)

        return True

    def save_prefs(self):
        import json

        if not os.path.isdir(self.prefs_folder):
            try:
                os.makedirs(self.prefs_folder)
            except:
                self.log('unable to create folder %s' % self.prefs_folder)
                return False

        prefix = self.prefs_folder + os.path.sep + self.bundle_name
        prefs_file_path = prefix + '.' + self.flame_user_name + '.' + self.flame_project_name + '.prefs.json'
        prefs_user_file_path = prefix + '.' + self.flame_user_name  + '.prefs.json'
        prefs_global_file_path = prefix + '.prefs.json'

        try:
            with open(prefs_file_path, 'w') as prefs_file:
                json.dump(self.prefs, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_file_path)
            self.log_debug(e)

        try:
            with open(prefs_user_file_path, 'w') as prefs_file:
                json.dump(self.prefs_user, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_user_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_user, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_user_file_path)
            self.log_debug(e)

        try:
            with open(prefs_global_file_path, 'w') as prefs_file:
                json.dump(self.prefs_global, prefs_file, indent=4)
            self.log_debug('preferences saved to %s' % prefs_global_file_path)
            self.log_debug('preferences contents:\n' + json.dumps(self.prefs_global, indent=4))
        except Exception as e:
            self.log('unable to save preferences to %s' % prefs_global_file_path)
            self.log_debug(e)
            
        return True

    def check_bundle_id(self):
        bundle_id_file_path = os.path.join(
            os.path.dirname(self.site_packages_folder),
            'bundle_id'
            )

        bundle_id = self.version

        if (os.path.isdir(self.bundle_folder) and os.path.isfile(bundle_id_file_path)):
            self.log('checking existing bundle id %s' % bundle_id_file_path)
            try:
                with open(bundle_id_file_path, 'r') as bundle_id_file:
                    if bundle_id_file.read() == bundle_id:
                        self.log('bundle folder exists with id matching current version')
                        bundle_id_file.close()
                        return True
                    else:
                        self.log('existing env bundle id does not match current one')
                        return False
            except Exception as e:
                self.log(pformat(e))
                return False
        elif not os.path.isdir(self.bundle_folder):
            self.log('bundle folder does not exist: %s' % self.bundle_folder)
            return False
        elif not os.path.isfile(bundle_id_file_path):
            self.log('bundle id file does not exist: %s' % bundle_id_file_path)
            return False

    def unpack_bundle(self, bundle_path):
        start = time.time()
        script_file_name, ext = os.path.splitext(os.path.realpath(__file__))
        script_file_name += '.py'
        # self.log('script file: %s' % script_file_name)
        script = None
        payload = None

        try:
            with open(script_file_name, 'r+') as scriptfile:
                script = scriptfile.read()
                start_position = script.rfind('# bundle payload starts here')
                
                if script[start_position -1: start_position] != '\n':
                    scriptfile.close()
                    return False

                start_position += 33
                payload = script[start_position:-4]
                # scriptfile.truncate(start_position - 34)
                scriptfile.close()
        except Exception as e:
            self.log_exception(e)
            return False
        
        del script
        if not payload:
            return False
        
        if payload.startswith('BUNDLE_PAYLO'):
            self.log(f'No payload attached to {__file__}')
            self.log('Nothing to unpack')
            return False

        bundle_backup_folder = ''
        if os.path.isdir(bundle_path):
            bundle_backup_folder = os.path.realpath(bundle_path + '.previous')
            if os.path.isdir(bundle_backup_folder):
                try:
                    cmd = 'rm -rf "' + os.path.realpath(bundle_backup_folder) + '"'
                    self.log('removing previous backup folder')
                    self.log('Executing command: %s' % cmd)
                    os.system(cmd)
                except Exception as e:
                    self.log_exception(e)
                    return False
            try:
                cmd = 'mv "' + os.path.realpath(bundle_path) + '" "' + bundle_backup_folder + '"'
                self.log('backing up existing bundle folder')
                self.log('Executing command: %s' % cmd)
                os.system(cmd)
            except Exception as e:
                self.log_exception(e)
                return False

        try:
            self.log('creating new bundle folder: %s' % bundle_path)
            os.makedirs(bundle_path)
        except Exception as e:
            self.log_exception(e)
            return False

        payload_dest = os.path.join(
            bundle_path, 
            self.sanitize_name(self.bundle_name + '.' + __version__ + '.bundle.tar.gz')
            )
        
        try:
            import base64
            self.log('unpacking payload: %s' % payload_dest)
            with open(payload_dest, 'wb') as payload_file:
                payload_file.write(base64.b64decode(payload))
                payload_file.close()
            cmd = 'tar xf "' + payload_dest + '" -C "' + bundle_path + '/"'
            self.log('Executing command: %s' % cmd)
            status = os.system(cmd)
            self.log('exit status %s' % os.WEXITSTATUS(status))

            # self.log('cleaning up %s' % payload_dest, logfile)
            # os.remove(payload_dest)
        
        except Exception as e:
            self.log_exception(e)
            return False

        delta = time.time() - start
        self.log('bundle extracted to %s' % bundle_path)
        self.log('extracting bundle took %s sec' % '{:.1f}'.format(delta))

        del payload
        try:
            os.remove(payload_dest)
        except Exception as e:
            self.log_exception(e)

        try:
            with open(os.path.join(bundle_path, 'bundle_id'), 'w') as bundle_id_file:
                bundle_id_file.write(self.version)
        except Exception as e:
            self.log_exception(e)
            return False
        
        return True

    def log_exception(self, e):
        self.log(pformat(e))
        self.log_debug(pformat(traceback.format_exc()))

    def sanitize_name(self, name_to_sanitize):
        import re
        if name_to_sanitize is None:
            return None
        
        stripped_name = name_to_sanitize.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        result = exp.sub('_', stripped_name)
        return re.sub('_\_+', '_', result)

    def sanitized(self, text):
        import re

        if text is None:
            return None
        
        text = text.strip()
        exp = re.compile(u'[^\w\.-]', re.UNICODE)

        if isinstance(text, str):
            result = exp.sub('_', text)
        else:
            decoded = text.decode('utf-8')
            result = exp.sub('_', decoded).encode('utf-8')

        return re.sub('_\_+', '_', result)

    def create_timestamp_uid(self):
        import random
        import uuid
        from datetime import datetime

        def number_to_letter(number):
            # Map each digit to a letter
            mapping = {
                '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
                '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'
            }
            return ''.join(mapping.get(char, char) for char in number)

        uid = ((str(uuid.uuid4()).replace('-', '')).upper())
        uid = ''.join(random.sample(number_to_letter(uid), 4))
        timestamp = (datetime.now()).strftime('%Y%b%d_%H%M').upper()
        return f'{timestamp}_{uid}'

    def normalize_values(self, image_array, torch = None):
        if torch is None:
            import torch

        def custom_bend(x):
            linear_part = x
            exp_bend = torch.sign(x) * torch.pow(torch.abs(x), 1 / 4 )
            return torch.where(x > 1, exp_bend, torch.where(x < -1, exp_bend, linear_part))

        # transfer (0.0 - 1.0) onto (-1.0 - 1.0) for tanh
        image_array = (image_array * 2) - 1
        # bend values below -1.0 and above 1.0 exponentially so they are not larger then (-4.0 - 4.0)
        image_array = custom_bend(image_array)
        # bend everything to fit -1.0 - 1.0 with hyperbolic tanhent
        image_array = torch.tanh(image_array)
        # move it to 0.0 - 1.0 range
        image_array = (image_array + 1) / 2

        return image_array
    
    def restore_normalized_values(self, image_array, torch = None):
        if torch is None:
            import torch

        def custom_de_bend(x):
            linear_part = x
            exp_deband = torch.sign(x) * torch.pow(torch.abs(x), 4 )
            return torch.where(x > 1, exp_deband, torch.where(x < -1, exp_deband, linear_part))

        epsilon = torch.tensor(4e-8, dtype=torch.float32).to(image_array.device)
        # clamp image befor arctanh
        image_array = torch.clamp((image_array * 2) - 1, -1.0 + epsilon, 1.0 - epsilon)
        # restore values from tanh  s-curve
        image_array = torch.arctanh(image_array)
        # restore custom bended values
        image_array = custom_de_bend(image_array)
        # move it to 0.0 - 1.0 range
        image_array = ( image_array + 1.0) / 2.0

        return image_array

    def check_requirements(self, requirements):
        sys.path_importer_cache.clear()

        def import_required_packages(requirements, cleanup = False):
            import re

            packages_by_name = {re.split(r'[!<>=]', req)[0]: req for req in requirements}
            missing_requirements = []

            for package_name in packages_by_name.keys():
                # try:
                #    self.message_queue.put(
                #        {'type': 'info', 'message': f'Checking requirements... importing {package_name}'}
                #    )
                # except:
                #    pass
                try:
                    sys.path_importer_cache.clear()                   
                    __import__(package_name)
                    
                    if cleanup:
                        if package_name in sys.modules:
                            del sys.modules[package_name]
                            sys.path_importer_cache.clear()                   

                    # try:
                    #    self.message_queue.put(
                    #        {'type': 'info', 'message': f'Checking requirements... successfully imported {package_name}'}
                    #    )
                    # except:
                    #    pass
                except:
                    missing_requirements.append(packages_by_name.get(package_name))
            return missing_requirements
        
        missing_requirements = import_required_packages(requirements)

        if missing_requirements:
            # try to add bundled packafe folder into sys.path and check if it is possible to import
            if not self.site_packages_folder in sys.path:
                sys.path.append(self.site_packages_folder)
            missing_requirements = import_required_packages(requirements, cleanup = False)
            
            # missing_requirements = import_required_packages(requirements, cleanup = True)
            # cleanup sys path and import cache afterwards
            # if self.site_packages_folder in sys.path:
            #    sys.path.remove(self.site_packages_folder)
            # sys.path_importer_cache.clear()

            return missing_requirements
        else:
            return []

    def wt_create_temp_library(self, selection):        
        try:
            import flame

            clip = selection[0]
            temp_library_name = self.app_name + '_' + self.fw.sanitized(clip.name.get_value()) + '_' + self.fw.create_timestamp_uid()
            self.temp_library_name = temp_library_name
            self.temp_library = flame.projects.current_project.create_shared_library(temp_library_name)
            flame.execute_shortcut('Save Project')
            flame.projects.current_project.refresh_shared_libraries()
            return self.temp_library
        
        except Exception as e:
            message_string = f'Unable to create temp shared library:\n"{e}"'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': self.close_application}
            )
            return None

    def wt_create_destination_node(self, clip, num_frames):
        try:
            import flame
            import numpy as np

            model_name = self.model_state_dict.get('model_name', 'UnknownModel')
            destination_node_name = clip.name.get_value() + f'_{model_name}_ML'
            self.app_state['destination_node_name'] = destination_node_name
            destination_node_id = ''
            server_handle = WireTapServerHandle('localhost')
            clip_node_id = clip.get_wiretap_node_id()
            clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
            fmt = WireTapClipFormat()
            if not clip_node_handle.getClipFormat(fmt):
                raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
            bits_per_channel = fmt.bitsPerPixel() // fmt.numChannels()
            self.bits_per_channel = bits_per_channel
            self.format_tag = fmt.formatTag()
            self.fmt = fmt

            self.temp_library.release_exclusive_access()
            node_id = self.temp_library.get_wiretap_node_id()
            parent_node_handle = WireTapNodeHandle(server_handle, node_id)
            destination_node_handle = WireTapNodeHandle()

            if not parent_node_handle.createClipNode(
                destination_node_name,  # display name
                fmt,  # clip format
                "CLIP",  # extended (server-specific) type
                destination_node_handle,  # created node returned here
            ):
                raise Exception(
                    "Unable to create clip node: %s." % parent_node_handle.lastError()
                )

            if not destination_node_handle.setNumFrames(int(num_frames)):
                raise Exception(
                    "Unable to set the number of frames: %s." % clip_node_handle.lastError()
                )
            
            dest_fmt = WireTapClipFormat()
            if not destination_node_handle.getClipFormat(dest_fmt):
                raise Exception(
                    "Unable to obtain clip format: %s." % clip_node_handle.lastError()
                )
            
            # '''
            metadata = dest_fmt.metaData()
            metadata_tag = dest_fmt.metaDataTag()
            metadata = metadata.replace('<ProxyFormat>default</ProxyFormat>', '<ProxyFormat>none</ProxyFormat>')
            destination_node_handle.setMetaData(metadata_tag, metadata)
            # '''

            destination_node_id = destination_node_handle.getNodeId().id()

        except Exception as e:
            message_string = f'Error creating destination wiretap node:\n {e}'
            self.message_queue.put(
                {'type': 'mbox',
                'message': message_string,
                'action': None}
            )
            return None
        finally:
            server_handle = None
            clip_node_handle = None
            parent_node_handle = None
            destination_node_handle = None

        return destination_node_id
    
    def wt_delete_destination_node(self, destination_node_id):
        server_handle = WireTapServerHandle('localhost')
        clip_node_handle = WireTapNodeHandle(server_handle, destination_node_id)
        clip_node_handle.destroyNode()
        server_handle = None
        clip_node_handle = None

    def wt_read_image_data_torch(self, clip, frame_number):
        import flame
        import numpy as np
        import torch

        try:
            server_handle = WireTapServerHandle('localhost')
            clip_node_id = clip.get_wiretap_node_id()
            clip_node_handle = WireTapNodeHandle(server_handle, clip_node_id)
            fmt = WireTapClipFormat()
            if not clip_node_handle.getClipFormat(fmt):
                raise Exception('Unable to obtain clip format: %s.' % clip_node_handle.lastError())
            num_frames = WireTapInt()
            if not clip_node_handle.getNumFrames(num_frames):
                raise Exception(
                    "Unable to obtain number of frames: %s." % clip_node_handle.lastError()
                )

            buff = "0" * fmt.frameBufferSize()

            if not clip_node_handle.readFrame(int(frame_number), buff, fmt.frameBufferSize()):
                raise Exception(
                    '[read_image_data] Unable to obtain read frame %i: %s.' % (frame_number, clip_node_handle.lastError())
                )
            
            frame_buffer_size = fmt.frameBufferSize()
            
            bits_per_channel = fmt.bitsPerPixel() // fmt.numChannels()

            if bits_per_channel == 8:
                buff_tail = frame_buffer_size - (fmt.height() * fmt.width() * fmt.numChannels())
                np_image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=np.uint8)[:-1 * buff_tail]
                image_array = torch.from_numpy(np_image_array.copy())
                del np_image_array
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array / 255

            elif bits_per_channel == 10:
                dt = np.uint16
                byte_array = np.frombuffer(bytes(buff, 'latin-1'), dtype='>u4')
                # byte_array = np.frombuffer(bytes(buff, 'latin-1'), dtype='<u4')
                values_10bit = np.empty((len(byte_array) * fmt.numChannels(),), dtype=np.uint16)
                values_10bit[::3] = (byte_array >> 22) & 0x3FF
                values_10bit[1::3] = (byte_array >> 12) & 0x3FF
                values_10bit[2::3] = (byte_array >> 2) & 0x3FF
                image_array = torch.from_numpy(values_10bit.astype(np.float32))
                image_array = image_array[:fmt.height() * fmt.width() * fmt.numChannels()]
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array / 1024

            elif bits_per_channel == 16 and not('float' in fmt.formatTag()):
                dt = np.uint16
                buff_tail = (frame_buffer_size // np.dtype(dt).itemsize) - (fmt.height() * fmt.width() * fmt.numChannels())
                image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=dt)[:-1 * buff_tail]
                image_array = torch.from_numpy(image_array.astype(np.float32))
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array / 65535
            
            elif (bits_per_channel == 16) and ('float' in fmt.formatTag()):
                buff_tail = (frame_buffer_size // np.dtype(np.float16).itemsize) - (fmt.height() * fmt.width() * fmt.numChannels())
                np_image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=np.float16)[:-1 * buff_tail]
                image_array = torch.from_numpy(np_image_array.copy())
                del np_image_array
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array

            elif bits_per_channel == 32:
                buff_tail = (frame_buffer_size // np.dtype(np.float32).itemsize) - (fmt.height() * fmt.width() * fmt.numChannels())
                np_image_array = np.frombuffer(bytes(buff, 'latin-1'), dtype=np.float32)[:-1 * buff_tail]
                image_array = torch.from_numpy(np_image_array.copy())
                del np_image_array
                image_array = image_array.to(
                    device = self.torch_device,
                    dtype = torch.float32,
                    non_blocking=True
                    )
                image_array = image_array.reshape((fmt.height(), fmt.width(),  fmt.numChannels()))
                image_array = torch.flip(image_array, [0])
                return image_array

            else:
                raise Exception('Unknown image format')
            
        except Exception as e:
            self.message_queue.put(
                {'type': 'mbox',
                'message': f'Unable to read source image: {e}',
                'action': None}
            )

        finally:
            server_handle = None
            clip_node_handle = None

    def read_openexr_file(self, file_path, header_only = False):
        """
        Reads data from an OpenEXR file specified by the file path.

        This function opens an OpenEXR file and reads its contents, either the header information only or the full data, including image data. It utilizes the MinExrReader to process the file.

        Parameters:
        - file_path (str): Path to the OpenEXR file to be read.
        - header_only (bool, optional): If True, only header information is read. Defaults to False.

        Returns:
        - dict: A dictionary containing the OpenEXR file's metadata and image data (if header_only is False). The dictionary includes the following keys:
            - 'attrs': Attributes of the OpenEXR file.
            - 'compr': Compression type used in the OpenEXR file.
            - 'channel_names': Names of the channels in the OpenEXR file.
            - 'channel_types': Data types of the channels in the OpenEXR file.
            - 'shape': The shape of the image data, rearranged as (height, width, channels).
            - 'image_data': Numpy array of the image data if header_only is False. The data is transposed to match the shape (height, width, channels).

        Note:
        - The function uses a context manager to ensure the file is properly closed after reading.
        - It assumes the existence of a class method `MinExrReader` for reading the OpenEXR file.
        """

        import numpy as np
        with open(file_path, 'rb') as sfp:
            source_reader = self.MinExrReader(sfp, header_only)
            result = {
                'attrs': source_reader.attrs,
                'compr': source_reader.compr,
                'channel_names': source_reader.channel_names,
                'channel_types': source_reader.channel_types,
                'shape': (source_reader.shape[0], source_reader.shape[2], source_reader.shape[1]),
            }
            if not header_only:
                result['image_data'] = source_reader.image.transpose(0, 2, 1)[:, :, ::-1].copy()
            del source_reader
        return result
