PROJECT_DIR="/home/zhangyz/expert-kit"

function cleanup_ek_env() {
    unset LIBTORCH
    unset DYLD_FALLBACK_LIBRARY_PATH
    unset LD_LIBRARY_PATH
    unset EK_CONFIG
    unset DS_TINY_ROOT
    unset QWEN3_30B_A3B_ROOT
    deactivate 
}

cd() {
    builtin cd "$@" || return
    CURRENT_DIR=$(realpath .)
    
    if [[ "$PREV_DIR" == "$PROJECT_DIR"* && "$CURRENT_DIR" != "$PROJECT_DIR"* ]]; then
        cleanup_ek_env
    fi
    
    if [[ "$CURRENT_DIR" == "$PROJECT_DIR"* ]]; then
        export LIBTORCH=$(realpath "$PROJECT_DIR/vendor/libtorch")
        export DYLD_FALLBACK_LIBRARY_PATH=$(realpath "$PROJECT_DIR/vendor/libtorch/lib")
        export LD_LIBRARY_PATH=$(realpath "$PROJECT_DIR/vendor/libtorch/lib")
        export EK_CONFIG=$(realpath "$PROJECT_DIR/dev/hello-world.config.yaml")
        export DS_TINY_ROOT=$(realpath "$PROJECT_DIR/ek-db/resources/ds-tiny/")
        export QWEN3_30B_A3B_ROOT=$(realpath -s "$PROJECT_DIR/qwen3-30b-a3b") # ln -s to /data/...
        source "$PROJECT_DIR/.venv/bin/activate"

        export HTTP_PROXY=http://127.0.0.1:7890
        export HTTPS_PROXY=http://127.0.0.1:7890
        export NO_PROXY=localhost,127.0.0.1,192.168.*,10.*
        ulimit -n 524288
    fi
    PREV_DIR=$CURRENT_DIR
}

PREV_DIR=$(realpath .)