from asyncio.coroutines import iscoroutine
import asyncio
import threading


def run_coroutine(coroutine):
    if iscoroutine(coroutine):
        asyncio.run_coroutine_threadsafe(coroutine, asyncio.get_event_loop())
