"""Canonical variable metadata and alias registration.

The registry separates stable public quantity names from historical Bifrost/IDL
aliases and on-disk storage names.  Human-readable descriptions are mandatory
for official package variables and also serve as runtime documentation for
quantity getter functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

Getter = Callable[..., object]


@dataclass(frozen=True)
class Variable:
    """Minimal metadata needed by the Bifrost variable resolver."""

    location: str | None = None
    unit_type: str | None = None
    component: str | None = None
    getter: Getter | None = None
    storage_name: str | None = None
    description: str | None = None


class VariableRegistry(dict):
    """Dictionary-like registry with canonical names and lightweight aliases."""

    def __init__(self, *args, **kwargs):
        """Create a registry, copying aliases when cloning another registry."""
        super().__init__(*args, **kwargs)
        self.aliases: dict[str, str] = {}
        if args and isinstance(args[0], VariableRegistry):
            self.aliases.update(args[0].aliases)

    def resolve(self, name: str) -> str:
        """Return the canonical registry name for *name*."""
        seen = set()
        while name in self.aliases:
            if name in seen:
                raise RuntimeError(f"Alias cycle detected for {name!r}")
            seen.add(name)
            name = self.aliases[name]
        return name

    def get(self, name, default=None):
        """Dictionary ``get`` with alias resolution."""
        return super().get(self.resolve(name), default)

    def __getitem__(self, name):
        """Return metadata for a canonical name or alias."""
        return super().__getitem__(self.resolve(name))

    def register_alias(self, alias: str, canonical: str):
        """Register *alias* for an existing canonical quantity."""
        canonical = self.resolve(canonical)
        if alias == canonical:
            return
        if alias in self and self.resolve(alias) != canonical:
            raise ValueError(f"Alias {alias!r} conflicts with an existing variable")
        if alias in self.aliases and self.resolve(alias) != canonical:
            raise ValueError(f"Alias {alias!r} already points to another variable")
        self.aliases[alias] = canonical

    def register(
        self,
        name: str,
        *,
        location: str | None = None,
        unit_type: str | None = None,
        component: str | None = None,
        getter: Getter | None = None,
        storage_name: str | None = None,
        description: str | None = None,
        aliases: Iterable[str] = (),
    ) -> Variable:
        """Create or update one canonical variable specification."""
        name = self.resolve(name)
        old = super().get(name, Variable())
        spec = Variable(
            location=location if location is not None else old.location,
            unit_type=unit_type if unit_type is not None else old.unit_type,
            component=component if component is not None else old.component,
            getter=getter if getter is not None else old.getter,
            storage_name=storage_name if storage_name is not None else old.storage_name,
            description=description if description is not None else old.description,
        )
        super().__setitem__(name, spec)
        for alias in aliases:
            self.register_alias(alias, name)
        return spec


GLOBAL_REGISTRY = VariableRegistry()


def _require_description(name: str, description: str | None) -> str:
    """Validate and normalize mandatory official-variable descriptions."""
    if description is None or not str(description).strip():
        raise ValueError(f"Official variable {name!r} must define a non-empty description")
    return str(description).strip()


def register_variable(
    name: str,
    *,
    description: str,
    location: str | None = None,
    unit_type: str | None = None,
    component: str | None = None,
    storage_name: str | None = None,
    aliases=(),
):
    """Register official metadata and optionally decorate a derived getter.

    A human-readable description is mandatory for package variables so the
    registry remains self-documenting as the quantity catalog grows.
    """

    description = _require_description(name, description)
    GLOBAL_REGISTRY.register(
        name,
        location=location,
        unit_type=unit_type,
        component=component,
        storage_name=storage_name,
        description=description,
        aliases=aliases,
    )

    def decorator(func: Getter):
        """Attach the decorated resolver getter to the registered variable."""
        canonical = GLOBAL_REGISTRY.resolve(name)
        GLOBAL_REGISTRY[canonical] = replace(GLOBAL_REGISTRY[canonical], getter=func)
        if not func.__doc__:
            func.__doc__ = (
                f"Resolver getter for ``{canonical}``.\n\n"
                f"{description}\n\n"
                "The resolver supplies native Bifrost orientation/units and "
                "applies public orientation, units, and selections afterwards."
            )
        return func

    return decorator


def register_vector(
    names,
    *,
    locations,
    unit_type: str,
    description: str | None = None,
    descriptions=None,
    storage_names=None,
    aliases=None,
):
    """Register the x/y/z components of an official vector.

    ``names`` may either be a base name (``"b"`` -> ``b_x/b_y/b_z``) or an
    explicit three-name sequence.  Give either one base ``description`` or
    three component-specific ``descriptions``.
    """

    if isinstance(names, str):
        names = tuple(f"{names}_{component}" for component in "xyz")
    else:
        names = tuple(names)
    locations = tuple(locations)

    if len(names) != 3 or len(locations) != 3:
        raise ValueError("register_vector expects three names and three locations")

    if descriptions is None:
        base = _require_description("/".join(names), description)
        base = base.rstrip(".")
        descriptions = tuple(
            f"{component.upper()} component of {base[0].lower() + base[1:]}."
            for component in "xyz"
        )
    else:
        descriptions = tuple(descriptions)
        if len(descriptions) != 3:
            raise ValueError("descriptions must contain three entries")
        descriptions = tuple(
            _require_description(name, desc)
            for name, desc in zip(names, descriptions)
        )

    if storage_names is None:
        storage_names = (None, None, None)
    else:
        storage_names = tuple(storage_names)
        if len(storage_names) != 3:
            raise ValueError("storage_names must contain three entries")

    if aliases is None:
        aliases = ((), (), ())
    else:
        if len(aliases) != 3:
            raise ValueError("aliases must contain three component entries")
        aliases = tuple(
            (item,) if isinstance(item, str) else tuple(item)
            for item in aliases
        )

    for name, location, component, storage_name, component_aliases, component_description in zip(
        names, locations, "xyz", storage_names, aliases, descriptions
    ):
        GLOBAL_REGISTRY.register(
            name,
            location=location,
            unit_type=unit_type,
            component=component,
            storage_name=storage_name,
            description=component_description,
            aliases=component_aliases,
        )

    return names
