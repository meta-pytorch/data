#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

AWS_ENABLED=1

# shellcheck disable=SC2086
pip install ${PYTORCH_PIP_PREFIX:-} torchdata --extra-index-url "${PYTORCH_PIP_DOWNLOAD_URL}"

# Fixes https://github.com/meta-pytorch/data/issues/1539
pip install packaging


case "${AWS_ENABLED}" in
    "0")
        python ./test/smoke_test/smoke_test.py --no-s3
        ;;
    "1")
        python ./test/smoke_test/smoke_test.py
        ;;
    *)
        exit 1
        ;;
esac
