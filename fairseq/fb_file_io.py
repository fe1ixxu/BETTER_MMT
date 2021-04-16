#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__file__)


def update_path_manager(IOPathManager):
    try:
        from iopath.fb.manifold import ManifoldPathHandler

        IOPathManager.register_handler(ManifoldPathHandler())
    except KeyError:
        logging.debug("ManifoldPathHandler already registered.")
    except ImportError:
        logging.debug(
            "ManifoldPathHandler couldn't be imported. Maybe missing fb-only files."
        )
