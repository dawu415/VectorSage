from sseclient import SSEClient
import codecs
import requests

class SSERequestClient(SSEClient): # Method is either POST or GET
    def __init__(self, url, post_payload, method="POST", last_id=None, retry=3000, session=None, chunk_size=1024, **kwargs):
        self.method = method
        self.post_payload = post_payload
        super().__init__(url, last_id, retry, session, chunk_size, **kwargs)

    def _connect(self):
        if self.last_id:
            self.requests_kwargs['headers']['Last-Event-ID'] = self.last_id

        # Use session if set.  Otherwise fall back to requests module.
        requester = self.session or requests
        if self.method == 'POST':
            self.resp = requester.post(self.url, stream=True,data=self.post_payload, **self.requests_kwargs)
        else:
            self.resp = requester.get(self.url, stream=True, **self.requests_kwargs)
        self.resp_iterator = self.iter_content()
        encoding = self.resp.encoding or self.resp.apparent_encoding
        self.decoder = codecs.getincrementaldecoder(encoding)(errors='replace')

        # TODO: Ensure we're handling redirects.  Might also stick the 'origin'
        # attribute on Events like the Javascript spec requires.
        self.resp.raise_for_status()

    def close(self):
        self.resp.close()
