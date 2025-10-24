FROM ghcr.io/spack/tutorial:icpp25

USER spack

RUN git clone --depth=2  https://github.com/spack/spack

# Initialize Spack env for subsequent commands
RUN . spack/share/spack/setup-env.sh && \
    spack bootstrap now && \
    spack compiler find && \
    spack mirror add tutorial /mirror && \
    spack mirror add develop https://binaries.spack.io/develop && \
    spack buildcache keys --install --trust && \
    spack repo create $HOME/experiment experiment && \
    spack repo add $HOME/experiment/spack_repo/experiment && \
    spack config add config:url_fetch_method:'curl -k -q'

ENV PATH="/home/spack/spack/bin:/opt/spack/view/bin:/root/spack/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
COPY --chown=spack:spack --chmod=0755 container/entrypoint.sh /usr/local/bin/entrypoint.sh
WORKDIR /opt/spack-it
RUN mkdir data
COPY --chown=spack:spack --chmod=0644 data/buildsys.pkl data/
RUN mkdir extraction
COPY --chown=spack:spack --chmod=0644 --chown=spack:spack --chmod=0644  extraction/package.py extraction/package_schema.py extraction/
RUN mkdir generate
COPY --chown=spack:spack --chmod=0644 --chown=spack:spack --chmod=0644 generate/eval.py generate/
COPY --chown=spack:spack --chmod=0644 --chown=spack:spack --chmod=0644 container/scores.py .

WORKDIR /home/spack
ENV PYTHONPATH="/opt/spack-it"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/sleep","infinity"]   # absolute path is important
