#pragma once
#include <utility>

template<typename Iterable>
class EnumerateObject
{
private:
	Iterable _iter;
	std::size_t _size;
	decltype(std::begin(_iter)) _begin;
	const decltype(std::end(_iter)) _end;

public:
	EnumerateObject(Iterable iter) :
		_iter(iter),
		_size(0),
		_begin(std::begin(iter)),
		_end(std::end(iter))
	{}

	const EnumerateObject& begin() const { return *this; }
	const EnumerateObject& end()   const { return *this; }

	bool operator!=(const EnumerateObject&) const
	{
		return _begin != _end;
	}

	void operator++()
	{
		++_begin;
		++_size;
	}

	auto operator*() const
		-> std::pair<std::size_t, decltype(*_begin)>
	{
		return { _size, *_begin };
	}
};

