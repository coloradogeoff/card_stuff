import Foundation

final class DirectoryWatcher {
    private var source: DispatchSourceFileSystemObject?
    private var fd: Int32 = -1
    var onChange: (() -> Void)?

    func watch(url: URL) {
        stop()
        fd = open(url.path, O_EVTONLY)
        guard fd >= 0 else { return }
        source = DispatchSource.makeFileSystemObjectSource(fileDescriptor: fd, eventMask: .write, queue: .main)
        source?.setEventHandler { [weak self] in
            self?.onChange?()
        }
        source?.setCancelHandler { [weak self] in
            if let fd = self?.fd, fd >= 0 { close(fd) }
        }
        source?.resume()
    }

    func stop() {
        source?.cancel()
        source = nil
    }

    deinit { stop() }
}
