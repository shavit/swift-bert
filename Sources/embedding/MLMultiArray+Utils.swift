import CoreML
import Accelerate

extension MLMultiArray {
    func toArray<T: Numeric>() -> Array<T> {
        let stride = MemoryLayout<T>.stride
        let allocated = UnsafeMutableRawBufferPointer.allocate(byteCount: self.count * stride, alignment: MemoryLayout<T>.alignment)
        return self.withUnsafeBytes { ptr in
            memcpy(allocated.baseAddress!, ptr.baseAddress!, self.count * stride)
            let start = allocated.bindMemory(to: T.self).baseAddress!
            return Array<T>(UnsafeBufferPointer(start: start, count: self.count))
        }
    }
}

extension MLMultiArray {
    var floats: [Float]? {
        guard self.dataType == .float32 else { return nil }
        var result: [Float] = Array(repeating: 0, count: self.count)
        return self.withUnsafeBytes { ptr in
            guard let source = ptr.baseAddress else { return nil }
            result.withUnsafeMutableBytes { resultPtr in
                let dest = resultPtr.baseAddress!
                memcpy(dest, source, self.count * MemoryLayout<Float>.stride)
            }
            return result
        }

    }
}

extension MLMultiArray {
    static func +(lhs: MLMultiArray, rhs: MLMultiArray) -> MLMultiArray {
        assert(lhs.dataType == rhs.dataType && lhs.dataType == .float32)
        assert(lhs.shape.count == rhs.shape.count && lhs.shape[1].intValue == rhs.shape[1].intValue)

        let outShape: [NSNumber]
        let outLength: Int
        var ptr0: UnsafeMutablePointer<Float32>
        var ptr1: UnsafeMutablePointer<Float32>
        if lhs.shape[0].intValue >= rhs.shape[0].intValue {
            assert(rhs.shape[0].intValue == 1 || lhs.shape == rhs.shape) // A[m, n], B[1, n] || B[m, n]
            outShape = lhs.shape
            outLength = lhs.count
            ptr0 = UnsafeMutablePointer<Float32>(OpaquePointer(lhs.withUnsafeMutableBytes({ ptr, _ in ptr.baseAddress! })))
            ptr1 = UnsafeMutablePointer<Float32>(OpaquePointer(rhs.withUnsafeMutableBytes({ ptr, _ in ptr.baseAddress! })))
        } else {
            assert(lhs.shape[0].intValue == 1) // Swap when A[1, n], B[m, n]
            outShape = rhs.shape
            outLength = rhs.count
            ptr0 = UnsafeMutablePointer<Float32>(OpaquePointer(rhs.withUnsafeMutableBytes({ ptr, _ in ptr.baseAddress! })))
            ptr1 = UnsafeMutablePointer<Float32>(OpaquePointer(lhs.withUnsafeMutableBytes({ ptr, _ in ptr.baseAddress! })))
        }

        let output = try! MLMultiArray(shape: outShape, dataType: .float32)
        var ptrOutput = UnsafeMutablePointer<Float32>(OpaquePointer(output.withUnsafeMutableBytes({ ptr, _ in ptr.baseAddress! })))
        vDSP_vadd(ptr0, 1, ptr1, 1, ptrOutput, 1, vDSP_Length(outLength))

        if lhs.shape[0].intValue != rhs.shape[0].intValue {
            for _ in 1..<outShape[0].intValue {
                ptr0 = ptr0.advanced(by: outShape[1].intValue)
                ptrOutput = ptrOutput.advanced(by: outShape[1].intValue)
                vDSP_vadd(ptr0, 1, ptr1, 1, ptrOutput, 1, vDSP_Length(outShape[1].intValue))
            }
        }

        return output
    }
}
