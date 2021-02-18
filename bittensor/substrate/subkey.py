# Python Subkey Wrapper
#
# Copyright 2018-2020 Stichting Polkascan (Polkascan Foundation).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shlex
import subprocess
from abc import ABC, abstractmethod

import docker
from docker.errors import ContainerError


class CommandFailException(Exception):
    pass


class InvalidConfigurationError(Exception):
    pass


class SubkeyImplementation(ABC):

    @abstractmethod
    def execute_command(self, command, stdin=None, json_output=True, **kwargs):
        pass

    def generate_key(self, network):
        return self.execute_command(['generate', '--network={}'.format(network)])

    def inspect(self, network, suri):
        return self.execute_command(['inspect', '--network={}'.format(network), suri])

    def vanity(self, network, pattern):
        return self.execute_command(['vanity', '--pattern={}'.format(pattern), '--network={}'.format(network)])

    def sign(self, data, suri, is_hex=True):

        return self.execute_command(
            command=['sign', '--hex', suri],
            stdin=data,
            json_output=False
        )


class DockerSubkeyImplementation(SubkeyImplementation):

    def __init__(self, docker_image=None):

        self.docker_image = docker_image or 'parity/subkey:latest'

    def execute_command(self, command, stdin=None, json_output=True, **kwargs):

        if json_output:
            command = command + ['--output-type=json']

        full_command = ' '.join([shlex.quote(el) for el in command])

        if stdin:
            full_command = '-c "echo -n \\"{}\\" | subkey {}"'.format(stdin, full_command)
        else:
            full_command = '-c "subkey {}"'.format(full_command)

        client = docker.from_env()
        try:
            output = client.containers.run(self.docker_image, full_command, entrypoint='/bin/sh')

            output = output[0:-1].decode()

            if json_output:
                output = json.loads(output[output.index('{'):])

            return output

        except ContainerError as e:
            raise CommandFailException('Docker Error: ', e)

        except json.JSONDecodeError as e:
            raise CommandFailException('Invalid format: ', e)


class LocalSubkeyImplementation(SubkeyImplementation):

    def __init__(self, subkey_path=None):
        self.subkey_path = subkey_path

    def execute_command(self, command, stdin=None, json_output=True, **kwargs):

        result = subprocess.run([self.subkey_path] + command + ['--output-type', 'json'], input=stdin, encoding='ascii',
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode > 0:
            raise CommandFailException(result.stderr)

        # Strip the newline in the end of the result
        output = result.stdout[0:-1]

        if json_output:
            output = json.loads(output)

        return output


class HttpSubkeyImplementation(SubkeyImplementation):

    def execute_command(self, command, stdin=None, json_output=True, **kwargs):
        pass


class Subkey:

    def __init__(self, use_docker=True, docker_image=None, subkey_path=None, subkey_host=None):

        if subkey_path:
            self.implementation = LocalSubkeyImplementation(subkey_path=subkey_path)
        elif subkey_host:
            self.implementation = HttpSubkeyImplementation()
        elif use_docker:
            self.implementation = DockerSubkeyImplementation(docker_image=docker_image)
        else:
            raise InvalidConfigurationError(
                'No valid subkey configuration, either set subkey_path, subkey_host or use_docker'
            )

    def execute_command(self, command):
        self.implementation.execute_command(command)

    def generate_key(self, network):
        return self.implementation.generate_key(network=network)

    def vanity(self, network, pattern):
        return self.implementation.vanity(network=network, pattern=pattern)

    def inspect(self, network, suri):
        return self.implementation.inspect(network=network, suri=suri)

    def sign(self, data, suri, is_hex=True):
        if is_hex:
            data = data.replace('0x', '')
        return self.implementation.sign(data=data, suri=suri, is_hex=is_hex)
