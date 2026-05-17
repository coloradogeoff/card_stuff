import Foundation

final class DirectoryWatcher {
    private var source: DispatchSourceFileSystemObject?
    private var fd: Int32 = -1
    var onChange: (() -> Void)?

    func watch(url: URL) {
        stop()
        let newFd = open(url.path, O_EVTONLY)
        fd = newFd
        guard fd >= 0 else { return }
        source = DispatchSource.makeFileSystemObjectSource(fileDescriptor: newFd, eventMask: .write, queue: .main)
        source?.setEventHandler { [weak self] in
            self?.onChange?()
        }
        source?.setCancelHandler {
            close(newFd)
        }
        source?.resume()
    }

    func stop() {
        source?.cancel()
        source = nil
    }

    deinit { stop() }
}
